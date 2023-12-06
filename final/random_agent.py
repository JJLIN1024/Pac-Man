import numpy as np
from ple import PLE
from ple.games.waterworld import WaterWorld
import time
import random
import math


class GreedyAgent():
    """
            This is our naive agent. It picks actions at random!
    """
    # game state
# {'player_velocity_y': 147.5536033780025,
#  'player_velocity_x': -46.34296875,
# --> creep dist is the distance of creep and player
#  'creep_dist': {'Y': [44.70139055098614, 163.65180113493955, 211.66231864803228],
#                  'R': [80.30614795692918, 145.85068519976804, 193.63647333370977],
#                  'G': [127.17360168557823, 43.97612192200753, 160.5704942056394]},
#  'step': 43,
#  'creep_pos':
#                  {'Y': [[151.8272534868923, 197.35765466620296], [121.04086432869069, 20.212341656788787], [44.879986179541305, 13.188012301622935]],
#                  'R': [[106.8457602181602, 172.04709132810382], [42.03919284756619, 155.00093032247742], [28.76817158895098, 58.49007671479184]],
#                  'G': [[142.57191303492797, 50.81412908937009], [143.15094045195514, 169.40603700808913], [50.34349271994585, 85.82152652852146]]},
# 'player_x': 187.12395833333335,
# 'player_y': 169.9285288293299}
#

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs, state):
        # ('1', 115)
        # ('2', 100)
        # ('3', 119)
        # ('4', 97)
        # ('5', None)
        # [('down', 115), ('right', 100), ('up', 119), ('left', 97)]
        px = state['player_x']
        py = state['player_y']
        vx = state['player_velocity_x']
        vy = state['player_velocity_y']
        time = state['step'] // 100
        Red = 5 * math.cos(time)
        Yellow = 5 * math.sin(time)
        sign = 1.0
        tmp = math.cos(time) + math.sin(time)
        if tmp < 0:
            sign = -1.0
        Green = 5 * sign * tmp ** 2 / 2
        # print(time, Red, Yellow, Green)
        if Red < 0 and Yellow < 0 and Green < 0:
            return None
        colors = {'Y': Yellow, 'R': Red, 'G': Green}
        # max_value_color = ['Yellow', 'Red', 'Green'][colors.index(max(colors))]
        min_distance = 9999999
        min_creep_index = None
        min_creep_pos = None

        for color, value in colors.items():
            if value > 0:
                distance = min(state['creep_dist'][color])
                if distance <= min_distance:
                    min_creep_index = state['creep_dist'][color].index(min(state['creep_dist'][color]))
                    min_creep_pos = state['creep_pos'][color][min_creep_index]

        # print('min_creep_pos', min_creep_pos)
        # print('player_pos', px, py)

        bestAction = None
        min_d = 99999999
        for action in self.actions:
            if action == 115:  # down
                new_vx, new_vy = vx * 0.975, (vy + 50) * 0.975
            elif action == 100:  # right
                new_vx, new_vy = (vx + 50) * 0.975, vy * 0.975
            elif action == 119:  # up
                new_vx, new_vy = vx * 0.975, (vy - 50) * 0.975
            elif action == 97:  # left
                new_vx, new_vy = (vx - 50) * 0.975, vy * 0.975
            elif action == None:
                new_vx, new_vy = vx * 0.975, vy * 0.975
            newX, newY = px + (new_vx * 3 / 30), py + (new_vy * 3 / 30)
            nextState = [newX, newY]
            # print(px, py)
            # print(nextState)
            new_distance = math.sqrt(math.pow(min_creep_pos[0] - nextState[0], 2) + math.pow(min_creep_pos[1] - nextState[1], 2))
            if new_distance < min_d:
                min_d = new_distance
                bestAction = action
        # print('bestAction', bestAction)
        return bestAction
        # return self.actions[np.random.randint(len(self.actions))]

# Don't display window.
# os.putenv('SDL_VIDEODRIVER', 'fbcon')
# os.environ["SDL_VIDEODRIVER"] = "dummy"


# ple.PLE(
#         game, fps=30,
#         frame_skip=1, num_steps=1,
#         reward_values={}, force_fps=True,
#         display_screen=False, add_noop_action=True,
#         NOOP=K_F15, state_preprocessor=None,
#         rng=24
#     )

# make a PLE instance.
# p is out environment
reward = 0.0
fps = 30
frame_skip = 1
num_steps = 1
reward_values = {}
force_fps = True  # slower speed
display_screen = False
add_noop_action = False
state_preprocessor = None
rng = 24

game = WaterWorld()  # create our game

# p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps, reward_values=reward_values, force_fps=force_fps, display_screen=display_screen, add_noop_action=add_noop_action, state_preprocessor=state_preprocessor, rng=rng)
p = PLE(game, force_fps=force_fps)

# For more repetitive results
random.seed(1)
np.random.seed(1)

# init agent and game.
p.init()
p.display_screen = True

# our Naive agent!
agent = GreedyAgent(p.getActionSet())

# start our loop
score = 0.0
score_list = []
for i in range(10):
    # if the game is over
    if p.game_over():
        p.reset_game()
    while p.game_over() == False:
        obs = p.getScreenRGB()
        state = p.getGameState()
        action = agent.pickAction(reward, obs, state)
        reward = p.act(action)  # reward after an action
    score = game.getScore()
    score_list.append(score)
    print "Trial no.{:02d} : score {:0.3f}".format(i, score)
print "average_score: {:02d}".format(sum(score_list) / len(score_list))

# Screen Shot
#     p.saveScreen("screen_capture.png")
