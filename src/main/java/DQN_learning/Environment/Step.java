package DQN_learning.Environment;

import DQN_learning.Learner.Learner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Step {
    private State beginState, endState;
    private int a;
    private double r;
    private boolean isTerminate;
    public Step(State begin, int action, double reward, State end, boolean isEnd){
        this.beginState = begin;
        this.a = action;
        this.r = reward;
        this.endState = end;
        this.isTerminate = isEnd;
    }
    public State getBeginState(){return beginState;}
    public State getEndState(){return endState;}
    public double[][][] getBeginStateConv(){return beginState.getConvVersion()[0];}
    public double[][][] getEndStateConv(){return endState.getConvVersion()[0];}
    public double[] getBeginStateDense(){return beginState.getState();}
    public double[] getEndStateDense(){return endState.getState();}
    public boolean isTerminate(){return isTerminate;}
    public INDArray getBeginState(Learner.InputType type) {
        if (type.equals(Learner.InputType.Convolution))
            return Nd4j.create(beginState.getConvVersion());
        else if (type.equals(Learner.InputType.Dense))
            return Nd4j.create(beginState.getState());
        else
            return Nd4j.zeros(0);
    }
    public INDArray getEndState(Learner.InputType type) {
        if (type.equals(Learner.InputType.Convolution))
            return Nd4j.create(endState.getConvVersion());
        else if (type.equals(Learner.InputType.Dense))
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
    public Step clone() {
        return new Step(beginState, a, r, endState, isTerminate);
    }
}
