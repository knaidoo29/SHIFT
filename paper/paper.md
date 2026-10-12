---
title: 'SHIFT: A lightweight, SciPy-based framework for distributed spectral transforms'
tags:
  - Python
  - Fourier transforms
  - MPI
  - parallel computing
  - scientific computing
  - spectral methods
authors:
  - name: Krishna Naidoo
    affiliation: 1
    corresponding: true
    # TODO: Add an ORCID if desired, and check whether other contributors
    # qualify for co-authorship and have agreed to be listed.
affiliations:
  - index: 1
    name: Institute of Cosmology and Gravitation, University of Portsmouth, United Kingdom
# Update to the actual manuscript date when submitting.
date: 12 October 2026
bibliography: paper.bib
---

# Summary

Fourier and related spectral transforms are widely used to analyse spatially
sampled data and solve numerical problems in physics and other computational
sciences. Although libraries for these transforms are well established,
integrating them into scientific workflows can require additional work to
construct frequency grids, maintain consistent conventions, and distribute
large arrays across multiple processors.

**SHIFT** (Scalable High-level Interface for Fourier Transforms) is a Python
library that brings together spectral transforms, Cartesian grid utilities,
and optional distributed-memory parallelism. SHIFT builds on the numerical
implementations provided by SciPy [@Virtanen2020], offering one-, two-, and
three-dimensional Fourier transforms together with discrete cosine and sine
transforms. With `mpi4py` [@Dalcin2021], these operations can be performed on
arrays partitioned across MPI processes using slab decomposition. The package
also provides reusable MPI communication routines for numerical and structured
NumPy arrays. SHIFT is intended to be lightweight to install and integrate,
without requiring users to separately install FFTW as a SHIFT dependency.

# Statement of need

Large numerical simulations and scientific-data analyses routinely use
transform-based methods to evaluate derivatives, solve Poisson equations,
filter fields, and estimate spatial statistics. In a single process,
`scipy.fft` supplies reliable transform implementations, but an application
must still coordinate data decomposition and communication to compute
transforms over grids that span multiple MPI ranks. A researcher may also
need consistent real-space and Fourier-space grids, normalization
conventions, and support for transforms associated with different boundary
conditions.

SHIFT was developed to reduce this implementation and deployment burden.
In particular, it provides MPI-enabled spectral transforms through an
interface written in Python, with SciPy as the transform backend. This avoids
requiring an additional FFTW installation and a package-specific FFTW-linked
extension, although MPI execution still requires an MPI runtime and
`mpi4py`. The target audience includes researchers developing numerical
solvers, simulation codes, and analysis pipelines who want familiar NumPy
arrays and straightforward integration with the scientific Python ecosystem.

# State of the field

SciPy provides an extensive collection of local transforms but does not, by
itself, provide an MPI-distributed FFT interface [@Virtanen2020]. `mpi4py`
provides direct access to MPI communication primitives, leaving the layout,
redistribution, and transform orchestration to application developers
[@Dalcin2021]. The `mpi4py-fft` package addresses parallel transforms more
directly, with an FFTW-based implementation and support for flexible
decompositions, including pencil decomposition [@Mortensen2019].

SHIFT serves a different design objective. Rather than replacing these
libraries or claiming a new FFT algorithm, it combines SciPy-backed
transforms with purpose-built Cartesian grid and MPI helpers in a package
that does not require an FFTW-specific build step. Implementing SHIFT as a
separate library allowed the dependency model, grid conventions, and
communication API to be developed together for reuse across scientific
programs. This choice also involves limitations: the current distributed
transforms use slab decomposition, and SHIFT should not be assumed to match
FFTW-based libraries in performance or scalability. A reproducible
comparison is needed before making quantitative claims about installation
reliability, runtime, or memory use.

# Software design

SHIFT separates local spectral operations from MPI redistribution. Local
transforms call `scipy.fft`, while the Cartesian utilities construct spatial
and corresponding transform-space grids. The interfaces retain explicit
physical box lengths and grid dimensions so that users can apply consistent
transform normalization and spectral operators. Alongside Fourier
transforms, the package provides discrete cosine and sine transforms, which
are useful in calculations with non-periodic boundary conditions.

For MPI-distributed two- and three-dimensional transforms, data begin in an
axis-aligned slab decomposition. SHIFT evaluates transforms along locally
complete axes, redistributes the slabs so another axis becomes local, and
then applies the remaining transform. The redistribution layer uses
`Alltoall` when partitions are equal and `Alltoallv` for unequal partitions.
The high-level transform code is kept separate from lower-level collective
communication and array-handling utilities. This separation enables the
same infrastructure to be reused outside the Fourier-transform routines.

The communication utilities can transfer arrays as contiguous bytes rather
than relying on MPI to infer a datatype for every NumPy structure. This
design accommodates structured records as well as ordinary numeric dtypes,
provided that communicating ranks agree on the memory layout. It does not
serialize Python objects or convert numerical values. SHIFT is implemented
in Python but relies on compiled SciPy and, when MPI is enabled, the
underlying MPI implementation through `mpi4py`.

The slab-decomposition design prioritizes a comparatively small interface
and ease of installation over the additional flexibility of pencil-based
parallel FFT libraries. The supported numerical operations are validated
with unit tests and transform round-trip checks; the repository also
includes MPI communication tests and continuous integration.

# Research impact statement

SHIFT is distributed as an open-source Python package, with public source
code, user documentation, versioned releases, and a software archive
[@NaidooSHIFT]. Its testing infrastructure checks transform correctness and
MPI communication across different numerical dtypes and array layouts.
These facilities make the package available for reuse in research code and
allow prospective users to independently evaluate core functionality.

<!-- TODO BEFORE SUBMISSION:
Provide stronger, specific public evidence of scholarly use or credible
near-term research significance. Suitable examples: (1) a publicly
reproducible Poisson-solver or power-spectrum analysis using SHIFT;
(2) an archived benchmark with commands, machine/MPI details, runtime and
memory measurements; (3) published research applications or public
third-party integrations when these can be disclosed. Do NOT describe
private downstream research repositories or results until that information
can be made public. JOSS reviewers assess real evidence, not merely
potential usefulness.
-->

# AI usage disclosure

OpenAI ChatGPT was used to assist in drafting this manuscript template.
<!-- TODO BEFORE SUBMISSION:
Expand this disclosure to describe accurately any AI-assisted software,
test, and documentation development, identify the tools/models and versions
used where possible, and describe how the human author independently
reviewed, edited, and validated the outputs. Confirm the final wording.
Do not claim that AI was unused.
-->

# Acknowledgements

<!-- TODO: List applicable funding and support, and verify grant numbers.
If none apply, use an appropriate short statement or omit only if permitted
by the journal's requirements. -->

# References
