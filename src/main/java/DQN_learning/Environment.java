package DQN_learning;

public abstract class Environment{
    public abstract Step performAction(int a);
    public abstract boolean isEnd();
    public abstract void reset();
    public abstract State getCurrentState();
}
