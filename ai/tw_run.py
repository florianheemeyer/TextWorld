'''
Created on 08.12.2018

@author: andre
'''

import textworld
import Agents
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("game")
    parser.add_argument("--episodes", type=int, default=20, metavar="EPISODES",
                        help="Number of games played")
    parser.add_argument("--steps", type=int, default=100, metavar="STEPS",
                        help="Number of steps per game played")
    parser.add_argument("--state-space", type=int, default=1000, metavar="STATE_SPACE",
                        help="Maximum number of possible states")
    parser.add_argument("--action-space", type=int, default=12, metavar="ACTION_SPACE",
                        help="Maximum number of possible actions in a step")
    parser.add_argument("--learning-rate", type=float, default=1, metavar="ALPHA",
                        help="Learning rate for new observations")
    parser.add_argument("--discount", type=float, default=0.99, metavar="GAMMA",
                        help="Discount factor for future rewards")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    env = textworld.start(args.game)  # Start an existing game.
    agent = Agents.SimpleReinforcementAgent(args.state_space, args.action_space, args.learning_rate, args.discount)
    
    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores = [], []
    env.enable_extra_info("description")
    env.enable_extra_info("inventory")
    for no_episode in range(args.episodes):
        print("Episode "+ str(no_episode))
        agent.reset(env)  # Tell the agent a new episode is starting.
        env.activate_state_tracking()
        env.compute_intermediate_reward()
        game_state = env.reset()  # Start new episode.
        reward = 0
        done = False
        
        
        for no_step in range(args.steps):
            command = agent.act(game_state, reward, done)
            #print(command)
            game_state, reward, done = env.step(command)
            #print(str(game_state))

            if done:
                break
        print(str(no_step) + " steps taken")
        print(str(len(agent.stateDictionary)) + " states explored in total")

        # See https://textworld-docs.maluuba.com/textworld.html#textworld.core.GameState
        avg_moves.append(game_state.nb_moves)
        avg_scores.append(game_state.score)
        agent.finish(game_state, reward, done)
        
    print("avg. steps: {:5.1f}; avg. score: {:4.1f} / 1.".format(sum(avg_moves)/args.episodes, sum(avg_scores)/args.episodes))
    
    #now we try to make it show what it learned

    agent.reset(env)  # Tell the agent a new episode is starting.
    env.activate_state_tracking()
    env.compute_intermediate_reward()
    game_state = env.reset()  # Start new episode.
    reward = 0
    done = False
    agent.pybrain_rlAgent._setLearning(False)
    #print(str(agent.pybrain_rlAgent.module))

    params = agent.pybrain_rlAgent.module.params.reshape(agent.pybrain_rlAgent.module.numRows,
                                                agent.pybrain_rlAgent.module.numColumns)
    
    for no_step in range(10):
            print(str(game_state))
            command = agent.act(game_state, reward, done)
            #print("Intermediate reward: " + str(game_state.intermediate_reward))
            game_state, reward, done = env.step(command)
            print("Command: " + str(command) + " (reward: " + str(game_state.intermediate_reward) + ")")
            if done:
                print(str(game_state))
                break
    
    
    env.close()








