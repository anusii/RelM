from differential_privacy.differential_privacy import laplace
import differential_privacy

arr = laplace(2, 2)
print(arr)
print(differential_privacy.samplers.laplace(2, 2))