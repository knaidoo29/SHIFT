coverage erase
coverage run -m pytest tests/normal_tests
mpirun -n 2 coverage run -m pytest tests/mpi_tests
coverage combine
coverage report -m