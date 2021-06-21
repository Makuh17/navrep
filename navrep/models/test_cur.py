from navrep.models.curiosity import *



buffer_test = ReplayBuffer(20,[1,3], [1,100])
print(buffer_test.get_newest_transition())

