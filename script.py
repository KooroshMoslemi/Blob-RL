import tensorflow as tf # tf 2.0.0 is used
import numpy as np
import gym
from BlobEnv import Blob , BlobEnv

# Optional for using gpu
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
assert(tf.test.is_gpu_available() == True)
# Optional for using gpu

learning_rate=1e-3
MAX_ITERS = 10000

class Memory:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self , new_observation , new_action , new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

def choose_action(model , observation , n_actions):
    observation = np.expand_dims(observation , axis = 0)

    logits = model.predict(observation)

    prob_weights = tf.nn.softmax(logits).numpy()

    action = np.random.choice(n_actions , size = 1 , p = prob_weights.flatten())[0]

    return action

def compute_loss(logits , actions , rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean( neg_logprob * rewards )
    return loss

def train_step(model , optimizer , observations , actions , discounted_rewards):
    with tf.GradientTape() as tape:
        logits = model(observations)
        loss = compute_loss(logits , actions , discounted_rewards)
    grads = tape.gradient(loss , model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def discount_rewards(rewards , gamma=0.95):
    discount_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0,len(rewards))):
        R = R * gamma + rewards[t]
        discount_rewards[t] = R
    return discount_rewards.astype(np.float32)

def create_blob_model(n_actions):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=64 , activation = 'relu'),
        tf.keras.layers.Dense(units=32 , activation = 'relu'),
        tf.keras.layers.Dense(units=n_actions , activation = None)
    ])
    return model

def run_blob(model_name):
    blob_t = tf.keras.models.load_model(f'saved_model/{model_name}')

    env = custom_init_env()
    obs = env.reset()
    done = False
    while not done:
        action = blob_t(np.expand_dims(obs, 0)).numpy().argmax()
        obs, reward, done = env.step(action)
        if(reward == env.FOOD_REWARD):print('won')
        elif(reward == -env.ENEMY_PENALTY):print('lost')
        env.render()

if __name__ == "__main__":

    #init
    env = BlobEnv()
    n_actions = env.ACTION_SPACE_SIZE
    blob_model = create_blob_model(n_actions)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    memory = Memory()


    #training loop
    for i_episode in range(MAX_ITERS):
        observation = env.reset()

        while True:

            action = choose_action(blob_model , observation ,n_actions)

            next_observation, reward, done = env.step(action)

            memory.add_to_memory(observation, action, reward)

            if done:
                total_reward = sum(memory.rewards)
                print(f"episode {i_episode+1}/{MAX_ITERS} reward: {total_reward}")
                train_step(blob_model, 
                     optimizer, 
                     observations = np.stack(memory.observations, 0), 
                     actions = np.array(memory.actions),
                     discounted_rewards = discount_rewards(memory.rewards))
                memory.clear()
                break

            observation = next_observation
    
    #save model
    blob_model.save('saved_model/blob_1000_6')

    #run
    #run_blob('blob_1000_5')