/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "opal/util/bit_ops.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/op/op.h"
#include "coll_offloaded.h"
#include "ompi/mca/coll/base/coll_base_topo.h"
#include "ompi/mca/coll/base/coll_base_util.h"

#include "../../../include/debugging_macros.h"


int ompi_coll_offloaded_allreduce_intra(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_offloaded_sendrecv_actual( const void* sendbuf, size_t scount, ompi_datatype_t* sdatatype, int dest, int stag, void* recvbuf, size_t rcount, ompi_datatype_t* rdatatype, int source, int rtag, struct ompi_communicator_t* comm, ompi_status_public_t* status );
/*
 *  allreduce_intra
 *
 *  Function:   - allreduce using other MPI collectives
 *  Accepts:    - same as MPI_Allreduce()
 *  Returns:    - MPI_SUCCESS or error code
 */
/*
 *   ompi_coll_base_allreduce_intra_recursivedoubling
 *
 *   Function:       Recursive doubling algorithm for allreduce operation
 *   Accepts:        Same as MPI_Allreduce()
 *   Returns:        MPI_SUCCESS or error code
 *
 *   Description:    Implements recursive doubling algorithm for allreduce.
 *                   Original (non-segmented) implementation is used in MPICH-2
 *                   for small and intermediate size messages.
 *                   The algorithm preserves order of operations so it can
 *                   be used both by commutative and non-commutative operations.
 *
 *         Example on 7 nodes:
 *         Initial state
 *         #      0       1      2       3      4       5      6
 *               [0]     [1]    [2]     [3]    [4]     [5]    [6]
 *         Initial adjustment step for non-power of two nodes.
 *         old rank      1              3              5      6
 *         new rank      0              1              2      3
 *                     [0+1]          [2+3]          [4+5]   [6]
 *         Step 1
 *         old rank      1              3              5      6
 *         new rank      0              1              2      3
 *                     [0+1+]         [0+1+]         [4+5+]  [4+5+]
 *                     [2+3+]         [2+3+]         [6   ]  [6   ]
 *         Step 2
 *         old rank      1              3              5      6
 *         new rank      0              1              2      3
 *                     [0+1+]         [0+1+]         [0+1+]  [0+1+]
 *                     [2+3+]         [2+3+]         [2+3+]  [2+3+]
 *                     [4+5+]         [4+5+]         [4+5+]  [4+5+]
 *                     [6   ]         [6   ]         [6   ]  [6   ]
 *         Final adjustment step for non-power of two nodes
 *         #      0       1      2       3      4       5      6
 *              [0+1+] [0+1+] [0+1+]  [0+1+] [0+1+]  [0+1+] [0+1+]
 *              [2+3+] [2+3+] [2+3+]  [2+3+] [2+3+]  [2+3+] [2+3+]
 *              [4+5+] [4+5+] [4+5+]  [4+5+] [4+5+]  [4+5+] [4+5+]
 *              [6   ] [6   ] [6   ]  [6   ] [6   ]  [6   ] [6   ]
 *
 */
int
ompi_coll_offloaded_allreduce_intra(const void *sbuf, void *rbuf,
                                    int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module)
{
    PRINT_DEBUG;
    int ret, line, rank, size, adjsize, remote, distance;
    int newrank, newremote, extra_ranks;
    char *tmpsend = NULL, *tmprecv = NULL, *tmpswap = NULL, *inplacebuf_free = NULL, *inplacebuf;
    ptrdiff_t span, gap = 0;


    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
            "coll:offloaded:allreduce_intra_recursivedoubling rank %d", rank));

    /* Special case for size == 1 */
    if (1 == size) {
        if (MPI_IN_PLACE != sbuf) {
            ret = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
            if (ret < 0) { line = __LINE__; goto error_hndl; }
        }
        return MPI_SUCCESS;
    }

    /* Allocate and initialize temporary send buffer */
    span = opal_datatype_span(&dtype->super, count, &gap);
    inplacebuf_free = (char*) malloc(span);
    if (NULL == inplacebuf_free) { ret = -1; line = __LINE__; goto error_hndl; }
    inplacebuf = inplacebuf_free - gap;

    if (MPI_IN_PLACE == sbuf) {
        ret = ompi_datatype_copy_content_same_ddt(dtype, count, inplacebuf, (char*)rbuf);
        if (ret < 0) { line = __LINE__; goto error_hndl; }
    } else {
        ret = ompi_datatype_copy_content_same_ddt(dtype, count, inplacebuf, (char*)sbuf);
        if (ret < 0) { line = __LINE__; goto error_hndl; }
    }

    tmpsend = (char*) inplacebuf;
    tmprecv = (char*) rbuf;

    /* Determine nearest power of two less than or equal to size */
    adjsize = opal_next_poweroftwo (size);
    adjsize >>= 1;

    /* Handle non-power-of-two case:
       - Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
       sets new rank to -1.
       - Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
       apply appropriate operation, and set new rank to rank/2
       - Everyone else sets rank to rank - extra_ranks
    */
    extra_ranks = size - adjsize;
    if (rank <  (2 * extra_ranks)) {
        if (0 == (rank % 2)) {
            ret = MCA_PML_CALL(send(tmpsend, count, dtype, (rank + 1),
                                    MCA_COLL_BASE_TAG_ALLREDUCE,
                                    MCA_PML_BASE_SEND_STANDARD, comm));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
            newrank = -1;
        } else {
            ret = MCA_PML_CALL(recv(tmprecv, count, dtype, (rank - 1),
                                    MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                    MPI_STATUS_IGNORE));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
            /* tmpsend = tmprecv (op) tmpsend */
            /*
             * TODO: replace this part with check for offloading capable function
             * */
            //ompi_op_reduce(op, tmprecv, tmpsend, count, dtype);
            newrank = rank >> 1;
        }
    } else {
        newrank = rank - extra_ranks;
    }

    /* Communication/Computation loop
       - Exchange message with remote node.
       - Perform appropriate operation taking in account order of operations:
       result = value (op) result
    */
    for (distance = 0x1; distance < adjsize; distance <<=1) {
        if (newrank < 0) break;
        /* Determine remote node */
        newremote = newrank ^ distance;
        remote = (newremote < extra_ranks)?
                 (newremote * 2 + 1):(newremote + extra_ranks);

        /* Exchange the data */
        ret = ompi_coll_offloaded_sendrecv_actual(tmpsend, count, dtype, remote,
                                                  MCA_COLL_BASE_TAG_ALLREDUCE,
                                                  tmprecv, count, dtype, remote,
                                                  MCA_COLL_BASE_TAG_ALLREDUCE,
                                                  comm, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

        /* Apply operation */
        if (rank < remote) {
             tmprecv = tmpsend (op) tmprecv
            /*
             * TODO: replace this part with check for offloading capable function
             * */
            ompi_op_reduce(op, tmpsend, tmprecv, count, dtype);
            tmpswap = tmprecv;
            tmprecv = tmpsend;
            tmpsend = tmpswap;
        } else {
            tmpsend = tmprecv (op) tmpsend;
            /*
            * TODO: MARKER - op to be offloaded
            * */
            ompi_op_reduce(op, tmprecv, tmpsend, count, dtype);
        }
    }

    /* Handle non-power-of-two case:
       - Odd ranks less than 2 * extra_ranks send result from tmpsend to
       (rank - 1)
       - Even ranks less than 2 * extra_ranks receive result from (rank + 1)
    */
    if (rank < (2 * extra_ranks)) {
        if (0 == (rank % 2)) {
            ret = MCA_PML_CALL(recv(rbuf, count, dtype, (rank + 1),
                                    MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                    MPI_STATUS_IGNORE));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
            tmpsend = (char*)rbuf;
        } else {
            ret = MCA_PML_CALL(send(tmpsend, count, dtype, (rank - 1),
                                    MCA_COLL_BASE_TAG_ALLREDUCE,
                                    MCA_PML_BASE_SEND_STANDARD, comm));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
        }
    }

    /* Ensure that the final result is in rbuf */
    if (tmpsend != rbuf) {
        ret = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, tmpsend);
        if (ret < 0) { line = __LINE__; goto error_hndl; }
    }

    if (NULL != inplacebuf_free) free(inplacebuf_free);
    return MPI_SUCCESS;

    error_hndl:
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "%s:%4d\tRank %d Error occurred %d\n",
            __FILE__, line, rank, ret));
    (void)line;  // silence compiler warning
    if (NULL != inplacebuf_free) free(inplacebuf_free);
    return ret;
}

int ompi_coll_offloaded_sendrecv_actual( const void* sendbuf, size_t scount,
                                         ompi_datatype_t* sdatatype,
                                         int dest, int stag,
                                         void* recvbuf, size_t rcount,
                                         ompi_datatype_t* rdatatype,
                                         int source, int rtag,
                                         struct ompi_communicator_t* comm,
                                         ompi_status_public_t* status )

{ /* post receive first, then send, then wait... should be fast (I hope) */
    PRINT_DEBUG;
    int err, line = 0;
    size_t rtypesize, stypesize;
    ompi_request_t *req;
    ompi_status_public_t rstatus;

    /* post new irecv */
    ompi_datatype_type_size(rdatatype, &rtypesize);
    //Go down into Libfabric RCV
    err = MCA_PML_CALL(irecv( recvbuf, rcount, rdatatype, source, rtag,
                              comm, &req));
    if (err != MPI_SUCCESS) { line = __LINE__; goto error_handler; }

    /* send data to children */
    ompi_datatype_type_size(sdatatype, &stypesize);

    //Go down into Libfabric SEND
    err = MCA_PML_CALL(send( sendbuf, scount, sdatatype, dest, stag,
                             MCA_PML_BASE_SEND_STANDARD, comm));
    if (err != MPI_SUCCESS) { line = __LINE__; goto error_handler; }

    err = ompi_request_wait( &req, &rstatus);
    if (err != MPI_SUCCESS) { line = __LINE__; goto error_handler; }

    if (MPI_STATUS_IGNORE != status) {
        *status = rstatus;
    }

    return (MPI_SUCCESS);

    error_handler:
    /* Error discovered during the posting of the irecv or send,
     * and no status is available.
     */
    OPAL_OUTPUT ((ompi_coll_base_framework.framework_output, "%s:%d: Error %d occurred\n",
            __FILE__, line, err));
    (void)line;  // silence compiler warning
    if (MPI_STATUS_IGNORE != status) {
        status->MPI_ERROR = err;
    }
    return (err);
}

