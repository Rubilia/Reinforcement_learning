package DQN_learning;

public class Step {
    private State beginStae, endState;
    private int a;
    private double r;
    private boolean isTerminate;
    public Step(State begin, int action, double reward, State end, boolean isEnd){
        this.beginStae = begin;
        this.a = action;
        this.r = reward;
        this.endState = end;
        this.isTerminate = isEnd;
    }
    public boolean isTerminate(){return isTerminate;}
    public State getBeginStae() {
        return beginStae;
    }
    public State getEndState() {
        return endState;
    }
    public int getA() {
        return a;
    }
    public double getR() {
        return r;
    }
}
