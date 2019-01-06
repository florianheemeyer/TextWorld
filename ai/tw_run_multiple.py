import tw_run
import shlex
import numpy as np

games = []
for i in range(10):
    games.append('../scripts/gen_games/v' + str(i+1) + '.ulx')

arguments = " --state-space 1000 --episodes 100 --action-space 150 --steps 200 --no-generated-command-check"

results = [(arguments, 0, False)]

for game in games:
    print("Playing game " + game)
    no_step, done = tw_run.main( shlex.split(game + arguments))
    results.append((game, no_step, done))

print(results)
np.save("results.npy", np.array(results))
