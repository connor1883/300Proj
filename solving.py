import numpy as np
import pandas as pd
import math

seed=4221724

def gen_data(n, y_min, alpha, mu, c, seed=1):
    
  # Validate parameters
    if y_min <= c:
        raise ValueError("y_min needs to be greater than c")
    if alpha <= 1:
        raise ValueError("alpha neds to be greater than 1")
    if mu <= 0:
        raise ValueError("mu needs to be positive")

    U = np.random.uniform(size=n)
    y = y_min * (1 - U) ** (-1.0 / alpha)

    U_exp = np.random.uniform(size=n)
    d = -mu * np.log(U_exp)

    y = np.sort(y)[::-1]
    d = np.sort(d)
    return y, d

def calc_prices_iterative(n, c, y, d):
    prices = np.zeros(n)
    prices[-1] = c  # Base case: price of the last agent is set to c
    
    # Calculate prices iteratively, starting from the second to last agent
    for j in range(n-2, -1, -1):  # Loop from n-2 down to 0
        theta_j = np.exp(-(d[j+1] - d[j]))
        prices[j] = theta_j * prices[j+1] + (1 - theta_j) * y[j]
        # Uncomment below for debugging
        # print(f"j: {j}, theta_j: {theta_j}, p_next: {prices[j+1]}, y_current: {y[j]}, p_current: {prices[j]}")
    
    return prices


def get_house_price(j, n, c, y, d):
    # Input checks:
    if j < 1 or j > n:
        raise ValueError(f"Invalid house index j={j}. j must be between 1 and {n}.")

    # Base case:
    if j == n:
        return c

    # Recursive step:
    # Compute theta_j
    theta_j = math.exp(-(d[j] - d[j-1]))  # Note: d is 0-indexed in Python, so d[j] corresponds to d_{j+1}.
    
    # Using 0-based indexing: y[j] is y_{j+1}
    p_j_plus_1 = get_house_price(j+1, n, c, y, d)
    
    p_j = theta_j * p_j_plus_1 + (1 - theta_j) * y[j]  # y[j] = y_{j+1} due to zero-index offset
    return p_j


def get_all_prices(n, c, y, d):
    # y and d are expected to be lists of length n.
    # y_1 = y[0], y_2 = y[1], ..., y_n = y[n-1]
    # d_1 = d[0], d_2 = d[1], ..., d_n = d[n-1]

    prices = []
    for j in range(1, n+1):
        p_j = get_house_price(j, n, c, y, d)
        prices.append(p_j)
    return prices



# testing
n = 2000
y_min = 80
alpha = 1.75
mu = 10
c = 50

y,d = gen_data(n,y_min,alpha,mu,c,seed)

#print(d[:5])  # Inspect the first 20 sorted distances

check = pd.DataFrame({
    'Income (y)': y,
    'Distance (d)': d,
})


print(check.head(10))

prices = calc_prices_iterative(n,c,y,d)

# Create DataFrame
test_df = pd.DataFrame({
    'Agent': range(1, n + 1),
    'Income (y)': y,
    'Distance (d)': d,
    'iterative': prices
})

print(test_df.head(2000))
