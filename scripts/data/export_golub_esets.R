#!/usr/bin/env Rscript

# Export the two serialized Bioconductor ExpressionSets to portable TSV files.
# This helper is kept beside the Golub download and transformation scripts.
# Accessing serialized slots through attributes avoids requiring Biobase solely
# for this one-time conversion.

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 3) {
    stop(
        "usage: export_golub_esets.R TRAIN_RDA TEST_RDA OUTPUT_DIR",
        call. = FALSE
    )
}

output_directory <- arguments[[3]]
if (!dir.exists(output_directory) &&
    !dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)) {
    stop(paste("cannot create output directory", output_directory), call. = FALSE)
}

extract_expression_set <- function(rdata_path, object_name, split_name) {
    environment <- new.env(parent = emptyenv())
    loaded <- suppressWarnings(load(rdata_path, envir = environment))
    if (!identical(loaded, object_name)) {
        stop(
            paste(
                rdata_path,
                "contains",
                paste(loaded, collapse = ", "),
                "instead of",
                object_name
            ),
            call. = FALSE
        )
    }

    expression_set <- get(object_name, envir = environment, inherits = FALSE)
    assay_data <- attr(expression_set, "assayData", exact = TRUE)
    if (!is.environment(assay_data) ||
        !exists("exprs", envir = assay_data, inherits = FALSE)) {
        stop(paste(object_name, "has no expression matrix"), call. = FALSE)
    }
    expression <- get("exprs", envir = assay_data, inherits = FALSE)
    phenotype_container <- attr(expression_set, "phenoData", exact = TRUE)
    phenotype <- attr(phenotype_container, "data", exact = TRUE)

    if (!is.matrix(expression) || !is.numeric(expression)) {
        stop(paste(object_name, "expression data are not a numeric matrix"), call. = FALSE)
    }
    if (!is.data.frame(phenotype)) {
        stop(paste(object_name, "phenotype data are not a data frame"), call. = FALSE)
    }
    if (is.null(rownames(expression)) || anyDuplicated(rownames(expression))) {
        stop(paste(object_name, "has invalid feature identifiers"), call. = FALSE)
    }
    required_columns <- c("Samples", "ALL.AML")
    if (!all(required_columns %in% colnames(phenotype))) {
        stop(paste(object_name, "is missing required phenotype columns"), call. = FALSE)
    }
    sample_ids <- as.character(phenotype$Samples)
    if (!identical(sample_ids, colnames(expression))) {
        stop(paste(object_name, "expression and phenotype samples differ"), call. = FALSE)
    }

    write.table(
        expression,
        file = file.path(output_directory, paste0(split_name, "_expression.tsv")),
        sep = "\t",
        quote = FALSE,
        col.names = NA
    )
    write.table(
        phenotype,
        file = file.path(output_directory, paste0(split_name, "_samples.tsv")),
        sep = "\t",
        quote = FALSE,
        row.names = FALSE,
        na = ""
    )
}

extract_expression_set(arguments[[1]], "Golub_Train", "train")
extract_expression_set(arguments[[2]], "Golub_Test", "test")
