package DQN_learning.Environment;

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

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof State))return false;
        State s = (State)o;
        if (state.length!=s.state.length)return false;
        for (int i = 0; i < state.length; i++) {
            if (s.state[i]!=state[i])return false;
        }
        return true;
    }
}
