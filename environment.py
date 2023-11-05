import gymnasium as gym
from gymnasium import spaces
import numpy as np


class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.max_steps = len(df.index)
        self.current_step = 0
        
        # Ação: 0 = não fazer nada, 1 = vender
        self.action_space = spaces.Discrete(2)

        # Definir o espaço de observação para incluir janela de preços dos últimos 30 dias
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(30*5,), dtype=np.float32)


    def reset(self, seed=None):
        # Inicializações adicionais ou redefinição de estados conforme necessário
        self.current_step = 30
        observation, info = self._next_observation(), {'current_step': self.current_step}
        # Cast the observation to the correct data type
        observation = observation.astype(np.float32)
        return observation, info
    
    def seed(self, seed=None):
        # Este método deve configurar a semente
        # Se você estiver usando NumPy ou qualquer outra biblioteca que utilize sementes, configure-as aqui
        np.random.seed(seed)

    # Certifique-se de que _next_observation retorne uma observação achatada se desejar remover o aviso.
    def _next_observation(self):
        # Select the window of prices for the last 30 days excluding the 'Data' column
        frame = self.df.iloc[self.current_step - 30:self.current_step, 1:].values
        # Flatten the frame
        frame = frame.flatten()
        # Convert the frame to float32 to match the observation space data type
        frame = frame.astype(np.float32)
        # The expected shape should be 30 days * 5 values per day
        expected_shape = (30 * 5,)
        assert frame.shape == expected_shape, f"Shape of flattened frame is {frame.shape}, expected {expected_shape}"
        return frame

    def step(self, action):
        self.current_step += 1

        if self.current_step > len(self.df.index):
            self.current_step = 0
        
        # Obter mudança de preço
        price_today = self.df.loc[self.current_step, '4. close']
        price_next_day = self.df.loc[self.current_step + 1, '4. close'] if self.current_step + 1 < self.max_steps else price_today
        price_change = price_next_day - price_today

        reward = self._calculate_reward(action, price_change)

        done = self.current_step >= self.max_steps - 1
        # Para ambientes sem uma condição de truncamento, você pode geralmente definir truncated como False.
        truncated = False
        # Você também pode querer adicionar informações adicionais no dicionário de info, mas isso é opcional.
        info = {}

        return self._next_observation(), reward, done, truncated, info

    def _calculate_reward(self, action, price_change):
        if action == 1:  # Ação de vender
            if price_change < 0:
                return 1  # Recompensa por vender antes do preço cair
            else:
                return -1  # Punido por vender antes do preço subir
        else:  # Ação de não fazer nada
            if price_change > 0:
                return 0.5  # Recompensado por manter enquanto o preço subia
            else:
                return -2  # Maior punição para evitar inação quando deveria vender

    def render(self, mode='human', close=False):
        # Esta implementação é simplificada e pode ser expandida para visualizar as negociações
        profit = self._calculate_profit()
        print(f'Step: {self.current_step}, Profit: {profit}')

    def _calculate_profit(self):
        # Este método é um stub para calcular o lucro do agente
        # Você pode expandi-lo para acompanhar as ações do agente e calcular o lucro real
        return 0
