/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
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
#include "ompi/mca/coll/base/base.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "coll_offloaded.h"

#include "../../../include/debugging_macros.h"

/*
 * Notes on evaluation rules and ordering
 *
 * The order is:
 *      use file based rules if presented (-coll_offloaded_dynamic_rules_filename = rules)
 * Else
 *      use forced rules (-coll_offloaded_dynamic_ALG_intra_algorithm = algorithm-number)
 * Else
 *      use fixed (compiled) rule set (or nested ifs)
 *
 */

/*
 *  allreduce_intra
 *
 *  Function:   - allreduce using other MPI collectives
 *  Accepts:    - same as MPI_Allreduce()
 *  Returns:    - MPI_SUCCESS or error code
 */
int
ompi_coll_offloaded_allreduce_intra_dec_dynamic (const void *sbuf, void *rbuf, int count,
                                             struct ompi_datatype_t *dtype,
                                             struct ompi_op_t *op,
                                             struct ompi_communicator_t *comm,
                                             mca_coll_base_module_t *module)
{
    PRINT_DEBUG;
    mca_coll_offloaded_module_t *offloaded_module = (mca_coll_offloaded_module_t*) module;

    OPAL_OUTPUT((ompi_coll_offloaded_stream, "ompi_coll_offloaded_allreduce_intra_dec_dynamic"));

    /* check to see if we have some filebased rules */
    if (offloaded_module->com_rules[ALLREDUCE]) {
        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int alg, faninout, segsize, ignoreme;
        size_t dsize;

        ompi_datatype_type_size (dtype, &dsize);
        dsize *= count;

        alg = ompi_coll_offloaded_get_target_method_params (offloaded_module->com_rules[ALLREDUCE],
                                                        dsize, &faninout, &segsize, &ignoreme);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
            return ompi_coll_offloaded_allreduce_intra_do_this (sbuf, rbuf, count, dtype, op,
                                                            comm, module,
                                                            alg, faninout, segsize);
        } /* found a method */
    } /*end if any com rules to check */

    if (offloaded_module->user_forced[ALLREDUCE].algorithm) {
        return ompi_coll_offloaded_allreduce_intra_do_this(sbuf, rbuf, count, dtype, op, comm, module,
                                                       offloaded_module->user_forced[ALLREDUCE].algorithm,
                                                       offloaded_module->user_forced[ALLREDUCE].tree_fanout,
                                                       offloaded_module->user_forced[ALLREDUCE].segsize);
    }
    return ompi_coll_offloaded_allreduce_intra_dec_fixed (sbuf, rbuf, count, dtype, op,
                                                      comm, module);
}

/*
 *    reduce_intra_dec
 *
 *    Function:    - seletects reduce algorithm to use
 *    Accepts:    - same arguments as MPI_reduce()
 *    Returns:    - MPI_SUCCESS or error code (passed from the reduce implementation)
 *
 */
int ompi_coll_offloaded_reduce_intra_dec_dynamic( const void *sbuf, void *rbuf,
                                              int count, struct ompi_datatype_t* dtype,
                                              struct ompi_op_t* op, int root,
                                              struct ompi_communicator_t* comm,
                                              mca_coll_base_module_t *module)
{
    PRINT_DEBUG;
    mca_coll_offloaded_module_t *offloaded_module = (mca_coll_offloaded_module_t*) module;

    OPAL_OUTPUT((ompi_coll_offloaded_stream, "coll:offloaded:reduce_intra_dec_dynamic"));

    /* check to see if we have some filebased rules */
    if (offloaded_module->com_rules[REDUCE]) {

        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int alg, faninout, segsize, max_requests;
        size_t dsize;

        ompi_datatype_type_size(dtype, &dsize);
        dsize *= count;

        alg = ompi_coll_offloaded_get_target_method_params (offloaded_module->com_rules[REDUCE],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
            return  ompi_coll_offloaded_reduce_intra_do_this (sbuf, rbuf, count, dtype,
                                                          op, root, comm, module,
                                                          alg, faninout,
                                                          segsize, max_requests);
        } /* found a method */
    } /*end if any com rules to check */

    if (offloaded_module->user_forced[REDUCE].algorithm) {
        return ompi_coll_offloaded_reduce_intra_do_this(sbuf, rbuf, count, dtype,
                                                    op, root, comm, module,
                                                    offloaded_module->user_forced[REDUCE].algorithm,
                                                    offloaded_module->user_forced[REDUCE].chain_fanout,
                                                    offloaded_module->user_forced[REDUCE].segsize,
                                                    offloaded_module->user_forced[REDUCE].max_requests);
    }
    return ompi_coll_offloaded_reduce_intra_dec_fixed (sbuf, rbuf, count, dtype,
                                                   op, root, comm, module);
}

