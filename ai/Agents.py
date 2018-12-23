'''
Created on 10.12.2018

@author: Peter_Enis
'''
from textworld import Agent
import pybrain.rl.agents
import pybrain.rl.learners.valuebased
import pybrain.rl.learners
import numpy as np
import pickle


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

    def __init__(self, sizeOfStateSpace, maxNumberOfActions, alpha, gamma):
        '''
        Constructor
        '''
        controller = pybrain.rl.learners.valuebased.ActionValueTable(sizeOfStateSpace,maxNumberOfActions)
        controller.initialize(0.5)
        learner = pybrain.rl.learners.Q(alpha, gamma)
        agent = pybrain.rl.agents.LearningAgent(controller, learner)
        self.pybrain_rlAgent = agent
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
            self.pybrain_rlAgent.giveReward(game_state.intermediate_reward)
        else:
            self.hasHistory = True

        legalActions = game_state.admissible_commands
        prunedActions = self.pruneAdmissibleCommands(legalActions)
        assert len(prunedActions) < self.pybrain_rlAgent.module.numColumns
        stateNumber = self.mapGameState(game_state,prunedActions)
        self.pybrain_rlAgent.integrateObservation(np.array([stateNumber]))

        while True:
            actionNumber = self.pybrain_rlAgent.getAction()
            if actionNumber < len(prunedActions):
                return prunedActions[int(actionNumber)]
            self.pybrain_rlAgent.giveReward(-1000)
            self.pybrain_rlAgent.integrateObservation(np.array([stateNumber]))


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
        self.pybrain_rlAgent.giveReward(game_state.intermediate_reward)
        self.pybrain_rlAgent.learn()
        self.pybrain_rlAgent.reset()
        self.hasHistory = False
    
    def mapGameState(self, game_state, prunedActions):
        sortedInventory = str(sorted([item for item in game_state.inventory.split("\n") if item != ""]))
        representation = game_state.description + sortedInventory
        if representation in self.stateDictionary:
            return self.stateDictionary[representation]
        else:
            value = len(self.stateDictionary)
            self.stateDictionary[representation] = value
            self.initGameStateValues(value,prunedActions)
            return value
        
    def pruneAdmissibleCommands(self, admissibleCommands):
        prunedCommands = list(filter(lambda x: ("drop" not in x) and ("examine" not in x) and ("look" not in x) and ("inventory" not in x), admissibleCommands))
        return prunedCommands
    
    def pruneAdmissibleCommandsLoosely(self, admissibleCommands):
        prunedCommands = list(filter(lambda x: ("examine" not in x) and ("look" not in x) and ("inventory" not in x), admissibleCommands))
        return prunedCommands

    def initGameStateValues(self,stateNumber,prunedActions):
        preferredVerbs = ["take", "open", "unlock"]

        for actionNumber in range(self.pybrain_rlAgent.module.numColumns):
            if actionNumber < len(prunedActions):
                verb = prunedActions[actionNumber].split()[0]
                if verb in preferredVerbs:
                    value = 0.7
                else:
                    value = 0.5
            else:
                value = -1000
            self.pybrain_rlAgent.module.updateValue(stateNumber, actionNumber, value)

    def saveState(self, filename):
        pickle.dump(self.pybrain_rlAgent.module.params, open( filename + "_t.p", "wb" ) )
        pickle.dump(self.stateDictionary, open( filename + "_d.p", "wb" ) )

    def loadState(self, filename):
        self.pybrain_rlAgent.module._params = pickle.load( open( filename + "_t.p", "rb" ) )
        self.stateDictionary = pickle.load( open( filename + "_d.p", "rb" ) )





class ReinforcementAgent2(Agent):
    '''
    classdocs
    '''

    def __init__(self, sizeOfStateSpace, maxNumberOfActions, alpha, gamma):
        '''
        Constructor
        '''
        controller = pybrain.rl.learners.valuebased.ActionValueTable(sizeOfStateSpace,maxNumberOfActions)
        controller.initialize(0.5)
        learner = pybrain.rl.learners.Q(alpha, gamma)
        agent = pybrain.rl.agents.LearningAgent(controller, learner)
        self.pybrain_rlAgent = agent
        self.stateDictionary = {}
        self.hasHistory = False
        self.lastAction = ""
        

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
        
        if "put" in self.lastAction or "drop" in self.lastAction or "insert" in self.lastAction:
            commandToUndo = self.lastAction.split()
            self.lastAction = ""
            
            #now undo last command
            if commandToUndo[0].lower() == "drop":
                cleanUpCommand = "take"
                for i in range(1, len(commandToUndo)):
                    cleanUpCommand += " " + commandToUndo[i]
                return cleanUpCommand
            elif commandToUndo[0].lower() == "put" or commandToUndo[0].lower() == "insert":
                cleanUpCommand = "take"
                for i in range(1, len(commandToUndo)):
                    if commandToUndo[i].lower() == "in" or commandToUndo[i].lower() == "into" or commandToUndo[i].lower() == "on" or commandToUndo[i].lower() == "to":
                        cleanUpCommand += " " + "from"
                    else:
                        cleanUpCommand += " " + commandToUndo[i]
                return cleanUpCommand
            assert False

        
        
        if self.hasHistory:
            self.pybrain_rlAgent.giveReward(game_state.intermediate_reward)
        else:
            self.hasHistory = True

        legalActions = game_state.admissible_commands
        prunedActions = self.pruneAdmissibleCommandsLoosely(legalActions)
        assert len(prunedActions) < self.pybrain_rlAgent.module.numColumns
        stateNumber = self.mapGameState(game_state,prunedActions)
        self.pybrain_rlAgent.integrateObservation(np.array([stateNumber]))

        while True:
            actionNumber = self.pybrain_rlAgent.getAction()
            if actionNumber < len(prunedActions):
                self.lastAction = prunedActions[int(actionNumber)]
                return prunedActions[int(actionNumber)]
            self.pybrain_rlAgent.giveReward(-1000)
            self.pybrain_rlAgent.integrateObservation(np.array([stateNumber]))


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
        self.pybrain_rlAgent.giveReward(game_state.intermediate_reward)
        self.pybrain_rlAgent.learn()
        self.pybrain_rlAgent.reset()
        self.hasHistory = False
        self.lastAction = ""

    
    def mapGameState(self, game_state, prunedActions):
        sortedInventory = str(sorted([item for item in game_state.inventory.split("\n") if item != ""]))
        representation = game_state.description + sortedInventory
        if representation in self.stateDictionary:
            return self.stateDictionary[representation]
        else:
            value = len(self.stateDictionary)
            self.stateDictionary[representation] = value
            self.initGameStateValues(value,prunedActions)
            return value
        
    def pruneAdmissibleCommands(self, admissibleCommands):
        prunedCommands = list(filter(lambda x: ("drop" not in x) and ("examine" not in x) and ("look" not in x) and ("inventory" not in x), admissibleCommands))
        return prunedCommands
    
    def pruneAdmissibleCommandsLoosely(self, admissibleCommands):
        prunedCommands = list(filter(lambda x: ("examine" not in x) and ("look" not in x) and ("inventory" not in x), admissibleCommands))
        return prunedCommands

    def initGameStateValues(self,stateNumber,prunedActions):
        preferredVerbs = ["take", "open", "unlock"]

        for actionNumber in range(self.pybrain_rlAgent.module.numColumns):
            if actionNumber < len(prunedActions):
                verb = prunedActions[actionNumber].split()[0]
                if verb in preferredVerbs:
                    value = 0.7
                else:
                    value = 0.5
            else:
                value = -1000
            self.pybrain_rlAgent.module.updateValue(stateNumber, actionNumber, value)

    def saveState(self, filename):
        pickle.dump(self.pybrain_rlAgent.module.params, open( filename + "_t.p", "wb" ) )
        pickle.dump(self.stateDictionary, open( filename + "_d.p", "wb" ) )

    def loadState(self, filename):
        self.pybrain_rlAgent.module._params = pickle.load( open( filename + "_t.p", "rb" ) )
        self.stateDictionary = pickle.load( open( filename + "_d.p", "rb" ) )


