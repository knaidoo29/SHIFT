coverage erase
coverage run -m pytest tests/normal_tests
mpirun -n 4 coverage run -m pytest tests/mpi_tests
coverage combine
coverage report -m