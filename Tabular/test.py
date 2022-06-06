from gym import spaces

action_space = spaces.Discrete(2)

action = 1
assert action_space.contains(action)