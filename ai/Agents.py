'''
Created on 10.12.2018

@author: andre
'''
from textworld import Agent
import pybrain.rl.agents
import pybrain.rl.learners.valuebased
import pybrain.rl.learners
import numpy as np


class HumanAgent(Agent):
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
        prunedCommands = list(filter(lambda x: ("drop" not in x) and ("examine" not in x) and ("look" not in x) and ("inventory" not in x), admissibleCommands))
        print("Legal Actions: " + str(len(admissibleCommands)) + " " + str(admissibleCommands))
        print("PrunedActions: " + str(len(prunedCommands)) + " " + str(prunedCommands))
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
    
    
class SimpleReinforcementAgent(Agent):
    '''
    classdocs
    '''

    def __init__(self, sizeOfStateSpace, maxNumberOfActions):
        '''
        Constructor
        '''
        controller = pybrain.rl.learners.valuebased.ActionValueTable(sizeOfStateSpace,maxNumberOfActions)
        controller.initialize(0.0)
        learner = pybrain.rl.learners.Q()
        agent = pybrain.rl.agents.LearningAgent(controller, learner)
        self.pybrain_rlAgent = agent
        self.currentIntermediateReward = 0.5
        self.stateDictionary = {}
        self.hasHistory = False
        

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
        if self.hasHistory:
            self.currentIntermediateReward += game_state.intermediate_reward
            #self.pybrain_rlAgent.giveReward(self.currentIntermediateReward)
            self.pybrain_rlAgent.giveReward(reward)
        else:
            self.hasHistory = True
        stateNumber = self.mapGameState(game_state)
        self.pybrain_rlAgent.integrateObservation(np.array([stateNumber]))
        legalActions = game_state.admissible_commands
        prunedActions = self.pruneAdmissibleCommands(legalActions)
        action = 0
                
        actionNumber = self.pybrain_rlAgent.getAction()
        if actionNumber < len(prunedActions):
            action = prunedActions[int(actionNumber)]
        else:
            action = "NoActionChosen"
        return action

        """
        print("Reward: " + str(reward))
        print("Done: " + str(done))
        print("Intermediate Reward: " + str(game_state.intermediate_reward))
        admissibleCommands = game_state.admissible_commands
        print("Legal Actions: " + str(len(admissibleCommands)) + " " + str(admissibleCommands))
        print(str(game_state))
        """

    def finish(self, game_state, reward: float, done: bool) -> None:
        """ Let the agent know the game has finished.

        Args:
            game_state: Game state at the moment the game finished.
            reward: Accumulated reward up until now.
            done: Whether the game has finished normally or not.
                If False, it means the agent's used up all of its actions.
        """
        self.pybrain_rlAgent.giveReward(reward)
        self.pybrain_rlAgent.learn()
        self.pybrain_rlAgent.reset()
        self.hasHistory = False
    
    def mapGameState(self, game_state):
        representation = game_state.description + game_state.inventory
        if representation in self.stateDictionary:
            return self.stateDictionary[representation]
        else:
            value = len(self.stateDictionary)
            self.stateDictionary[representation] = value
            return value
        
    def pruneAdmissibleCommands(self, admissibleCommands):
        prunedCommands = list(filter(lambda x: ("drop" not in x) and ("examine" not in x) and ("look" not in x) and ("inventory" not in x), admissibleCommands))
        return prunedCommands