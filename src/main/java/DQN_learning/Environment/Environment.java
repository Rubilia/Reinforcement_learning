package DQN_learning.Environment;

import DQN_learning.Learner.Learner;
import Tools.Pair;

public abstract class Environment{

    public abstract Step performAction(int a);
    public abstract boolean isEnd();
    public abstract void reset();
    public abstract State getCurrentState();
    public abstract Pair<String, Boolean> getScore(Learner dqn);
    @Override
    public abstract Environment clone();
}
