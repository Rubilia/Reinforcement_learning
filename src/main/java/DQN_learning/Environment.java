package DQN_learning;

public abstract class Environment{

    public abstract Step performAction(int a);
    public abstract boolean isEnd();
    public abstract void reset();
    public abstract State getCurrentState();
    public abstract String getScore(DQN_Learner dqn);
    @Override
    protected abstract Environment clone();
}
