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
#include <stdio.h>
#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_topo.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/op/op.h"
#include "coll_offloaded.h"

#include "../../../include/debugging_macros.h"

/*
 *  reduce_lin_intra
 *
 *  Function:   - reduction using O(N) algorithm
 *  Accepts:    - same as MPI_Reduce()
 *  Returns:    - MPI_SUCCESS or error code
 */
int
ompi_coll_offloaded_reduce_intra(const void *sbuf, void *rbuf, int count,
                                         struct ompi_datatype_t *dtype,
                                         struct ompi_op_t *op,
                                         int root,
                                         struct ompi_communicator_t *comm,
                                         mca_coll_base_module_t *module)
{
    int i, rank, err, size;
    ptrdiff_t extent, dsize, gap = 0;
    char *free_buffer = NULL;
    char *pml_buffer = NULL;
    char *inplace_temp_free = NULL;
    char *inbuf;

    /* Initialize */

    rank = ompi_comm_rank(comm);
    size = ompi_comm_size(comm);

    /* If not root, send data to the root. */

    if (rank != root) {
        err = MCA_PML_CALL(send(sbuf, count, dtype, root,
                                MCA_COLL_BASE_TAG_REDUCE,
                                MCA_PML_BASE_SEND_STANDARD, comm));
        return err;
    }

    dsize = opal_datatype_span(&dtype->super, count, &gap);
    ompi_datatype_type_extent(dtype, &extent);

    if (MPI_IN_PLACE == sbuf) {
        sbuf = rbuf;
        inplace_temp_free = (char*)malloc(dsize);
        if (NULL == inplace_temp_free) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        rbuf = inplace_temp_free - gap;
    }

    if (size > 1) {
        free_buffer = (char*)malloc(dsize);
        if (NULL == free_buffer) {
            if (NULL != inplace_temp_free) {
                free(inplace_temp_free);
            }
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        pml_buffer = free_buffer - gap;
    }

    /* Initialize the receive buffer. */

    if (rank == (size - 1)) {
        err = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
    } else {
        err = MCA_PML_CALL(recv(rbuf, count, dtype, size - 1,
                                MCA_COLL_BASE_TAG_REDUCE, comm,
                                MPI_STATUS_IGNORE));
    }
    if (MPI_SUCCESS != err) {
        if (NULL != free_buffer) {
            free(free_buffer);
        }
        return err;
    }

    /* Loop receiving and calling reduction function (C or Fortran). */

    for (i = size - 2; i >= 0; --i) {
        if (rank == i) {
            inbuf = (char*)sbuf;
        } else {
            err = MCA_PML_CALL(recv(pml_buffer, count, dtype, i,
                                    MCA_COLL_BASE_TAG_REDUCE, comm,
                                    MPI_STATUS_IGNORE));
            if (MPI_SUCCESS != err) {
                if (NULL != free_buffer) {
                    free(free_buffer);
                }
                return err;
            }

            inbuf = pml_buffer;
        }

        /* Perform the reduction */

        ompi_op_reduce(op, inbuf, rbuf, count, dtype);
    }

    if (NULL != inplace_temp_free) {
        err = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)sbuf, rbuf);
        free(inplace_temp_free);
    }
    if (NULL != free_buffer) {
        free(free_buffer);
    }

    /* All done */

    return MPI_SUCCESS;
}