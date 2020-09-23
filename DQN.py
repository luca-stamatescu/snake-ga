from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
import collections


class DQNAgent(object):
    def __init__(self, params):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.layer1 = params['layer1']
        self.layer2 = params['layer2']
        self.layer3 = params['layer3']
        self.layer4 = params['layer4']
        self.layer5 = params['layer5']
        self.layer6 = params['layer6']
        self.input_features=11

        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.model = self.network()

    def network(self):
        model = Sequential()
        model.add(Dense(output_dim=self.layer1, activation='relu', input_dim=self.input_features))
        model.add(Dense(output_dim=self.layer2, activation='relu'))
        model.add(Dense(output_dim=self.layer3, activation='relu'))
        # model.add(Dense(output_dim=self.layer4, activation='relu'))
        # model.add(Dense(output_dim=self.layer5, activation='relu'))
        # model.add(Dense(output_dim=self.layer6, activation='relu'))
        model.add(Dense(output_dim=3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model
    
    def get_state(self, game, player, food):

        import numpy as np
        from matplotlib import pyplot as plt
        import cv2
        import math
        px = 20
        x_width = 420
        y_height = 420
        vision_distance_x = 2
        vision_distance_y = 2

        if player.x_change == -20:
            direction = "left"
        if player.x_change == 20:
            direction = "right"
        if player.y_change == -20:
            direction = "up"
        if player.y_change == 20:
            direction = "down"

        snake_grid = np.zeros((int(y_height / px + 1), int(x_width / px + 1)))

        def make_dot(input_list, grid, px, value=1):
            """
            The first number in the list (of a point) is considered the X coordinate. For numpy matricies,
            the first number is the row index, which corresponds to "Y". For this function, we will continue to use the
            convention of (X,Y)
            """

            # test if it is a list containing multiple points to plot
            if any(isinstance(i, list) for i in input_list):
                for i in range(0, len(input_list)):
                    snake_grid[int(input_list[i][1] / px), int(input_list[i][0] / px)] = value
                return snake_grid

            # else it is only a single point:
            else:
                snake_grid[int(input_list[1] / px), int(input_list[0] / px)] = value
                return snake_grid

        border = []
        for i in range(0, int(y_height / px + 1)):
            border.append([0, i * px])
            border.append([int(x_width + 1), i * px])
            border.append([i * px, 0])
            border.append([i * px, int(y_height + 1)])
        make_dot(border, snake_grid, px);

        make_dot(player.position, snake_grid, px);

        make_dot(player.position[-1], snake_grid, px, 2);

        if direction == "right":
            rotated_snake_grid = np.rot90(snake_grid, 1)
            plt.imshow(rotated_snake_grid)
        if direction == "down":
            rotated_snake_grid = np.rot90(snake_grid, 2)
            plt.imshow(rotated_snake_grid)
        if direction == "left":
            rotated_snake_grid = np.rot90(snake_grid, 3)
            plt.imshow(rotated_snake_grid)
        if direction == "up":
            rotated_snake_grid = snake_grid
            plt.imshow(rotated_snake_grid)

        max_vision = max(vision_distance_x, vision_distance_x)
        rotated_snake_grid = np.pad(rotated_snake_grid, (max_vision, max_vision), 'constant', constant_values=(1, 1))

        # again, note the difference between X,Y and row,col. The slices have the buffer + 1 at the end, as numpy indexing is not inclusive.
        spx_rotated = np.where(rotated_snake_grid == 2)[1][0] * px
        spy_rotated = np.where(rotated_snake_grid == 2)[0][0] * px

        # again, note the difference between X,Y and row,col. The slices have the buffer + 1 at the end, as numpy indexing is not inclusive.
        vision = rotated_snake_grid[
                 int(spy_rotated / px - vision_distance_y):int(spy_rotated / px + vision_distance_y + 1),
                 int(spx_rotated / px - vision_distance_x):int(spx_rotated / px + vision_distance_x + 1)]
        state = []

        for i in range(0, vision.shape[0]):
            for j in range(0, vision.shape[1]):
                if i == math.floor(vision.shape[0] / 2) and j == math.floor(vision.shape[1] / 2):
                    print(i, j)
                    continue
                state.append(vision[i, j] == 1.0)

        danger_left = vision[1, 0]==1.0
        danger_right = vision[1, 2]==1.0
        danger_straight = vision[0, 1]==1.0
        state=[danger_left,danger_right,danger_straight]
        other_features = [


            player.x_change == -20,  # moved left
            player.x_change == 20,  # moved right
            player.y_change == -20,  # moved up
            player.y_change == 20,  # moved down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y  # food down
            ]

        for i in range(0, len(other_features)):
            state.append(other_features[i])


        test=1
        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)

    def set_reward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, self.input_features)))[0])
        target_f = self.model.predict(state.reshape((1, self.input_features)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, self.input_features)), target_f, epochs=1, verbose=0)