package DQN_learning;

import org.nd4j.linalg.api.ndarray.INDArray;

public class State{
    private INDArray state;
    public State(INDArray state){
        this.state = state;
    }
    public INDArray getState(){
        return this.state;
    }

    @Override
    public String toString() {
        return state.toString();
    }
}
