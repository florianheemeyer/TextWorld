'''
Created on 08.12.2018

@author: andre
'''

import textworld
import Agents

if __name__ == '__main__':
    env = textworld.start("/home/andre/TextWorld/gen_games/tw-game-vjs3cos0-house-GP-OgOJFl9Jtba5I1Rb.ulx")  # Start an existing game.
#    agent = Agents.HumanAgent()
    agent = Agents.SimpleReinforcementAgent()
    
    
    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores = [], []
    N = 1
    for no_episode in range(N):
        agent.reset(env)  # Tell the agent a new episode is starting.
        env.activate_state_tracking()
        env.compute_intermediate_reward()
        game_state = env.reset()  # Start new episode.
        

        reward = 0
        done = False
        for no_step in range(10):
            command = agent.act(game_state, reward, done)
            #print(command)
            game_state, reward, done = env.step(command)
            #print(str(game_state))

            if done:
                break

        # See https://textworld-docs.maluuba.com/textworld.html#textworld.core.GameState
        avg_moves.append(game_state.nb_moves)
        avg_scores.append(game_state.score)

    env.close()
    print("avg. steps: {:5.1f}; avg. score: {:4.1f} / 1.".format(sum(avg_moves)/N, sum(avg_scores)/N))