/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
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
 * Copyright (c) 2016      Intel, Inc.  All rights reserved.
 * Copyright (c) 2018      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_offloaded.h"

#include <stdio.h>

#include "mpi.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/mca/coll/base/coll_base_topo.h"
#include "coll_offloaded.h"
#include "coll_offloaded_dynamic_rules.h"
#include "coll_offloaded_dynamic_file.h"

#include "../../../include/debugging_macros.h"

static int offloaded_module_enable(mca_coll_base_module_t *module,
                   struct ompi_communicator_t *comm);
/*
 * Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.
 */
int ompi_coll_offloaded_init_query(bool enable_progress_threads,
                               bool enable_mpi_threads)
{
    PRINT_DEBUG;
    return OMPI_SUCCESS;
}


/*
 * Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 */
mca_coll_base_module_t *
ompi_coll_offloaded_comm_query(struct ompi_communicator_t *comm, int *priority)
{
    PRINT_DEBUG;
    mca_coll_offloaded_module_t *offloaded_module;

    OPAL_OUTPUT((ompi_coll_offloaded_stream, "coll:offloaded:module_offloaded query called"));

    /**
     * No support for inter-communicator yet.
     */
    if (OMPI_COMM_IS_INTER(comm)) {
        *priority = 0;
        return NULL;
    }

    /**
     * If it is inter-communicator and size is less than 2 we have specialized modules
     * to handle the intra collective communications.
     */
    if (OMPI_COMM_IS_INTRA(comm) && ompi_comm_size(comm) < 2) {
        *priority = 0;
        return NULL;
    }

    offloaded_module = OBJ_NEW(mca_coll_offloaded_module_t);
    if (NULL == offloaded_module) return NULL;

    *priority = ompi_coll_offloaded_priority;

    //Set function pointer to functions below
    offloaded_module->super.coll_module_enable = offloaded_module_enable;
    offloaded_module->super.ft_event = mca_coll_offloaded_ft_event;

    //Set function pointer to appropriate offloading function
    offloaded_module->super.coll_allreduce  = ompi_coll_offloaded_allreduce_intra;
    offloaded_module->super.coll_reduce     = ompi_coll_offloaded_reduce_intra;

    offloaded_module->super.coll_allgather  = NULL;
    offloaded_module->super.coll_allgatherv = NULL;
    offloaded_module->super.coll_alltoall   = NULL;
    offloaded_module->super.coll_alltoallv  = NULL;
    offloaded_module->super.coll_alltoallw  = NULL;
    offloaded_module->super.coll_barrier    = NULL;
    offloaded_module->super.coll_bcast      = NULL;
    offloaded_module->super.coll_exscan     = NULL;
    offloaded_module->super.coll_gather     = NULL;
    offloaded_module->super.coll_gatherv    = NULL;
    offloaded_module->super.coll_reduce_scatter = NULL;
    offloaded_module->super.coll_reduce_scatter_block = NULL;
    offloaded_module->super.coll_scan       = NULL;
    offloaded_module->super.coll_scatter    = NULL;
    offloaded_module->super.coll_scatterv   = NULL;
    return &(offloaded_module->super);
}

/*
 * Init module on the communicator
 */
static int
offloaded_module_enable( mca_coll_base_module_t *module,
                     struct ompi_communicator_t *comm )
{
    PRINT_DEBUG;
    int size;
    mca_coll_offloaded_module_t *offloaded_module = (mca_coll_offloaded_module_t *) module;
    mca_coll_base_comm_t *data = NULL;

    OPAL_OUTPUT((ompi_coll_offloaded_stream,"coll:offloaded:module_init called."));

    /* Allocate the data that hangs off the communicator */
    if (OMPI_COMM_IS_INTER(comm)) {
        size = ompi_comm_remote_size(comm);
    } else {
        size = ompi_comm_size(comm);
    }

    /* prepare the placeholder for the array of request* */
    data = OBJ_NEW(mca_coll_base_comm_t);
    if (NULL == data) {
        return OMPI_ERROR;
    }

    /* All done */
    offloaded_module->super.base_data = data;

    OPAL_OUTPUT((ompi_coll_offloaded_stream,"coll:offloaded:module_init offloaded is in use"));
    return OMPI_SUCCESS;
}

int mca_coll_offloaded_ft_event(int state) {
    PRINT_DEBUG;
    if(OPAL_CRS_CHECKPOINT == state) {
        ;
    }
    else if(OPAL_CRS_CONTINUE == state) {
        ;
    }
    else if(OPAL_CRS_RESTART == state) {
        ;
    }
    else if(OPAL_CRS_TERM == state ) {
        ;
    }
    else {
        ;
    }

    return OMPI_SUCCESS;
}
