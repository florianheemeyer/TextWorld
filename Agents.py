'''
Created on 10.12.2018

@author: andre
'''
import textworld
import pybrain.rl.agents
import pybrain.rl.learners.valuebased
import pybrain.rl.learners

class HumanAgent(textworld.Agent):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''

    def reset(self, env) -> None:
        """ Let the agent set some environment's flags.

        Args:
            env: TextWorld environment.
        """
        pass

    def act(self, game_state, reward: float, done: bool) -> str:
        """ Acts upon the current game state.

        Args:
            game_state: Current game state.
            reward: Accumulated reward up until now.
            done: Whether the game is finished.

        Returns:
            Text command to be performed in this current state.
        """
        print("Reward: " + str(reward))
        print("Done: " + str(done))
        print("Intermediate Reward: " + str(game_state.intermediate_reward))
        admissibleCommands = game_state.admissible_commands
        print("Legal Actions: " + str(len(admissibleCommands)) + " " + str(admissibleCommands))
        print(str(game_state))
        action = input()
        return str(action)
        

    def finish(self, game_state, reward: float, done: bool) -> None:
        """ Let the agent know the game has finished.

        Args:
            game_state: Game state at the moment the game finished.
            reward: Accumulated reward up until now.
            done: Whether the game has finished normally or not.
                If False, it means the agent's used up all of its actions.
        """
        pass
    
    
class SimpleReinforcementAgent(textworld.Agent):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        controller = pybrain.rl.learners.valuebased.ActionValueTable(10000,50)
        controller.initialize(1.)
        learner = pybrain.rl.learners.Q()
        agent = pybrain.rl.agents.LearningAgent(controller, learner)
        self.pybrain_rlAgent = agent
        

    def reset(self, env) -> None:
        """ Let the agent set some environment's flags.

        Args:
            env: TextWorld environment.
        """
        pass

    def act(self, game_state, reward: float, done: bool) -> str:
        """ Acts upon the current game state.

        Args:
            game_state: Current game state.
            reward: Accumulated reward up until now.
            done: Whether the game is finished.

        Returns:
            Text command to be performed in this current state.
        """
        print("Reward: " + str(reward))
        print("Done: " + str(done))
        print("Intermediate Reward: " + str(game_state.intermediate_reward))
        admissibleCommands = game_state.admissible_commands
        print("Legal Actions: " + str(len(admissibleCommands)) + " " + str(admissibleCommands))
        print(str(game_state))
        action = input()
        return str(action)
        

    def finish(self, game_state, reward: float, done: bool) -> None:
        """ Let the agent know the game has finished.

        Args:
            game_state: Game state at the moment the game finished.
            reward: Accumulated reward up until now.
            done: Whether the game has finished normally or not.
                If False, it means the agent's used up all of its actions.
        """
        pass