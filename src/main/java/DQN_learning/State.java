package DQN_learning;

public class State{
    private double[] state;
    private double[][][][] convVersion;
    public State(double[] state, double[][][][] conv){
        this.state = state;this.convVersion = conv;
    }
    public double[][][][] getConvVersion() {
        return convVersion;
    }
    public double[] getState(){
        return this.state;
    }
    @Override
    public String toString() {
        return state.toString();
    }
}
