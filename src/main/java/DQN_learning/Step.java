package DQN_learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
    public double[][][] getBeginStateConv(){return beginStae.getConvVersion()[0];}
    public double[][][] getEndStateConv(){return endState.getConvVersion()[0];}
    public double[] getBeginState(){return beginStae.getState();}
    public double[] getEndState(){return endState.getState();}
    public boolean isTerminate(){return isTerminate;}
    public INDArray getBeginState(DQN_Learner.InputType type) {
        if (type.equals(DQN_Learner.InputType.Covolution))
            return Nd4j.create(beginStae.getConvVersion());
        else if (type.equals(DQN_Learner.InputType.Dense))
            return Nd4j.create(beginStae.getState());
        else
            return Nd4j.zeros(0);
    }
    public INDArray getEndState(DQN_Learner.InputType type) {
        if (type.equals(DQN_Learner.InputType.Covolution))
            return Nd4j.create(endState.getConvVersion());
        else if (type.equals(DQN_Learner.InputType.Dense))
            return Nd4j.create(endState.getState());
        else
            return Nd4j.zeros(0);
    }
    public int getA() {
        return a;
    }
    public double getR() {
        return r;
    }
    @Override
    public String toString() {
        return "Action: " + a + ", reward: " + r + (isTerminate?"; Final state":"; Not final state");
    }
    @Override
    protected Step clone() {
        return new Step(beginStae, a, r, endState, isTerminate);
    }
}
