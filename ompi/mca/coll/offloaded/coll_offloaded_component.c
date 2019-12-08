/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
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
 * Copyright (c) 2008      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * These symbols are in a file by themselves to provide nice linker
 * semantics.  Since linkers generally pull in symbols by object
 * files, keeping these symbols as the only symbols in this file
 * prevents utility programs such as "ompi_info" from having to import
 * entire components just to query their version and parameters.
 */

#include "ompi_config.h"
#include "opal/util/output.h"
#include "coll_offloaded.h"
#include <stdio.h>
#include "mpi.h"
#include "ompi/mca/coll/coll.h"
#include "coll_offloaded.h"
#include "coll_offloaded_dynamic_file.h"

#include "../../../include/debugging_macros.h"


/*
 * Public string showing the coll ompi_offloaded component version number
 */
const char *ompi_coll_offloaded_component_version_string =
    "Open MPI offloaded collective MCA component version " OMPI_VERSION;

/*
 * Global variable
 */
int   ompi_coll_offloaded_stream = -1;
int   ompi_coll_offloaded_priority = 30;
bool  ompi_coll_offloaded_use_dynamic_rules = false;
char* ompi_coll_offloaded_dynamic_rules_filename = (char*) NULL;
int   ompi_coll_offloaded_init_tree_fanout = 4;
int   ompi_coll_offloaded_init_chain_fanout = 4;
int   ompi_coll_offloaded_init_max_requests = 128;
int   ompi_coll_offloaded_alltoall_small_msg = 200;
int   ompi_coll_offloaded_alltoall_intermediate_msg = 3000;

/* forced alogrithm variables */
/* indices for the MCA parameters */
coll_offloaded_force_algorithm_mca_param_indices_t ompi_coll_offloaded_forced_params[COLLCOUNT] = {{0}};
/* max algorithm values */
int ompi_coll_offloaded_forced_max_algorithms[COLLCOUNT] = {0};

/*
 * Local function
 */
static int offloaded_register(void);
static int offloaded_open(void);
static int offloaded_close(void);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */

mca_coll_offloaded_component_t mca_coll_offloaded_component = {
    /* First, fill in the super */
    {
        /* First, the mca_component_t struct containing meta information
           about the component itself */
        .collm_version = {
                MCA_COLL_BASE_VERSION_2_0_0,

            /* Component name and version */
            .mca_component_name = "offloaded",
            MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                                  OMPI_RELEASE_VERSION),

            /* Component open and close functions */
            .mca_open_component = offloaded_open,
            .mca_close_component = offloaded_close,
            .mca_register_component_params = offloaded_register,
        },
        .collm_data = {
            /* The component is checkpoint ready */
            MCA_BASE_METADATA_PARAM_CHECKPOINT
        },

        /* Initialization / querying functions */

        .collm_init_query = ompi_coll_offloaded_init_query,
        .collm_comm_query = ompi_coll_offloaded_comm_query,
    },

    /* priority of the module */
    0,

    /* offloaded component specific information */
    NULL /* ompi_coll_alg_rule_t ptr */
};

static int offloaded_register(void)
{
    PRINT_DEBUG;
    /* Use a low priority, but allow other components to be lower */
    ompi_coll_offloaded_priority = 30;
    (void) mca_base_component_var_register(&mca_coll_offloaded_component.super.collm_version,
                                           "priority", "Priority of the offloaded coll component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &ompi_coll_offloaded_priority);

    /* some initial guesses at topology parameters */
    ompi_coll_offloaded_init_tree_fanout = 4;
    (void) mca_base_component_var_register(&mca_coll_offloaded_component.super.collm_version,
                                           "init_tree_fanout",
                                           "Inital fanout used in the tree topologies for each communicator. This is only an initial guess, if a offloaded collective needs a different fanout for an operation, it build it dynamically. This parameter is only for the first guess and might save a little time",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &ompi_coll_offloaded_init_tree_fanout);

    ompi_coll_offloaded_init_chain_fanout = 4;
    (void) mca_base_component_var_register(&mca_coll_offloaded_component.super.collm_version,
                                           "init_chain_fanout",
                                           "Inital fanout used in the chain (fanout followed by pipeline) topologies for each communicator. This is only an initial guess, if a offloaded collective needs a different fanout for an operation, it build it dynamically. This parameter is only for the first guess and might save a little time",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &ompi_coll_offloaded_init_chain_fanout);

    ompi_coll_offloaded_alltoall_small_msg = 200;
    (void) mca_base_component_var_register(&mca_coll_offloaded_component.super.collm_version,
                                           "alltoall_small_msg",
                                           "threshold (if supported) to decide if small MSGs alltoall algorithm will be used",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &ompi_coll_offloaded_alltoall_small_msg);

    ompi_coll_offloaded_alltoall_intermediate_msg = 3000;
    (void) mca_base_component_var_register(&mca_coll_offloaded_component.super.collm_version,
                                           "alltoall_intermediate_msg",
                                           "threshold (if supported) to decide if intermediate MSGs alltoall algorithm will be used",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &ompi_coll_offloaded_alltoall_intermediate_msg);

    (void) mca_base_component_var_register(&mca_coll_offloaded_component.super.collm_version,
                                           "use_dynamic_rules",
                                           "Switch used to decide if we use static (compiled/if statements) or dynamic (built at runtime) decision function rules",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &ompi_coll_offloaded_use_dynamic_rules);

    ompi_coll_offloaded_dynamic_rules_filename = NULL;
    (void) mca_base_component_var_register(&mca_coll_offloaded_component.super.collm_version,
                                           "dynamic_rules_filename",
                                           "Filename of configuration file that contains the dynamic (@runtime) decision function rules",
                                           MCA_BASE_VAR_TYPE_STRING, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &ompi_coll_offloaded_dynamic_rules_filename);

    /* register forced params */
    ompi_coll_offloaded_allreduce_intra_check_forced_init(&ompi_coll_offloaded_forced_params[ALLREDUCE]);
    ompi_coll_offloaded_reduce_intra_check_forced_init(&ompi_coll_offloaded_forced_params[REDUCE]);

    return OMPI_SUCCESS;
}

static int offloaded_open(void)
{
    PRINT_DEBUG;
    int rc;

#if OPAL_ENABLE_DEBUG
    {
        int param;

        param = mca_base_var_find("ompi", "coll", "base", "verbose");
        if (param >= 0) {
            const int *verbose = NULL;
            mca_base_var_get_value(param, &verbose, NULL, NULL);
            if (verbose && verbose[0] > 0) {
                ompi_coll_offloaded_stream = opal_output_open(NULL);
            }
        }
    }
#endif  /* OPAL_ENABLE_DEBUG */

    /* now check that the user hasn't overrode any of the decision functions if dynamic rules are enabled */
    /* the user can redo this before every comm dup/create if they like */
    /* this is useful for benchmarking and user knows best tuning */
    /* as this is the component we only lookup the indicies of the mca params */
    /* the actual values are looked up during comm create via module init */

    /* intra functions first */
    /* if dynamic rules allowed then look up dynamic rules config filename, else we leave it an empty filename (NULL) */
    /* by default DISABLE dynamic rules and instead use fixed [if based] rules */
    if (ompi_coll_offloaded_use_dynamic_rules) {
        if( ompi_coll_offloaded_dynamic_rules_filename ) {
            OPAL_OUTPUT((ompi_coll_offloaded_stream,"coll:offloaded:component_open Reading collective rules file [%s]",
                         ompi_coll_offloaded_dynamic_rules_filename));
            rc = ompi_coll_offloaded_read_rules_config_file( ompi_coll_offloaded_dynamic_rules_filename,
                                                         &(mca_coll_offloaded_component.all_base_rules), COLLCOUNT);
            if( rc >= 0 ) {
                OPAL_OUTPUT((ompi_coll_offloaded_stream,"coll:offloaded:module_open Read %d valid rules\n", rc));
            } else {
                OPAL_OUTPUT((ompi_coll_offloaded_stream,"coll:offloaded:module_open Reading collective rules file failed\n"));
                mca_coll_offloaded_component.all_base_rules = NULL;
            }
        }
    }

    OPAL_OUTPUT((ompi_coll_offloaded_stream, "coll:offloaded:component_open: done!"));

    return OMPI_SUCCESS;
}

/* here we should clean up state stored on the component */
/* i.e. alg table and dynamic changable rules if allocated etc */
static int offloaded_close(void)
{
    PRINT_DEBUG;
    OPAL_OUTPUT((ompi_coll_offloaded_stream, "coll:offloaded:component_close: called"));

    /* dealloc alg table if allocated */
    /* dealloc dynamic changable rules if allocated */

    OPAL_OUTPUT((ompi_coll_offloaded_stream, "coll:offloaded:component_close: done!"));

    if( NULL != mca_coll_offloaded_component.all_base_rules ) {
        ompi_coll_offloaded_free_all_rules(mca_coll_offloaded_component.all_base_rules, COLLCOUNT);
        mca_coll_offloaded_component.all_base_rules = NULL;
    }

    return OMPI_SUCCESS;
}

static void
mca_coll_offloaded_module_construct(mca_coll_offloaded_module_t *module)
{
    PRINT_DEBUG;
    mca_coll_offloaded_module_t *offloaded_module = (mca_coll_offloaded_module_t*) module;

    for( int i = 0; i < COLLCOUNT; i++ ) {
        offloaded_module->user_forced[i].algorithm = 0;
        offloaded_module->com_rules[i] = NULL;
    }
}

OBJ_CLASS_INSTANCE(mca_coll_offloaded_module_t, mca_coll_base_module_t,
                   mca_coll_offloaded_module_construct, NULL);
