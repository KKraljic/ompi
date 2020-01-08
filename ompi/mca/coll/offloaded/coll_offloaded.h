/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2015 The University of Tennessee and The University
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

#ifndef MCA_COLL_OFFLOADED_EXPORT_H
#define MCA_COLL_OFFLOADED_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/mca/mca.h"
#include "ompi/request/request.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "opal/util/output.h"

BEGIN_C_DECLS
extern int   ompi_coll_offloaded_stream;


int ompi_coll_offloaded_init_query(bool enable_progress_threads,
                               bool enable_mpi_threads);

mca_coll_base_module_t *
ompi_coll_offloaded_comm_query(struct ompi_communicator_t *comm, int *priority);

int mca_coll_offloaded_ft_event(int state);

struct mca_coll_offloaded_component_t {
	/** Base coll component */
	mca_coll_base_component_2_0_0_t super;
	/** MCA parameter: Priority of this component */
	int offloaded_priority;
};
/**
 * Convenience typedef
 */
typedef struct mca_coll_offloaded_component_t mca_coll_offloaded_component_t;

/**
 * Global component instance
 */
OMPI_MODULE_DECLSPEC extern mca_coll_offloaded_component_t mca_coll_offloaded_component;

struct mca_coll_offloaded_module_t {
    mca_coll_base_module_t super;
};
typedef struct mca_coll_offloaded_module_t mca_coll_offloaded_module_t;

OBJ_CLASS_DECLARATION(mca_coll_offloaded_module_t);

#endif  /* MCA_COLL_offloaded_EXPORT_H */
