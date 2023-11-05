import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from environment import StockTradingEnv


df = pd.read_csv('stock.csv')

# Instancie o ambiente com o DataFrame
env = StockTradingEnv(df)

# Verifique se o ambiente está conforme esperado
check_env(env)

# Crie o agente
model = A2C('MlpPolicy', env, verbose=1)

# Treine o agente
model.learn(total_timesteps=10000)
