package DQN_learning.MonteCarlo;

import DQN_learning.Environment.Environment;
import DQN_learning.Learner.Learner;
import DQN_learning.Environment.State;
import DQN_learning.Environment.Step;
import Tools.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.*;

public class MC_Return {
    protected Random rnd = new Random();
    protected Environment environment;
    protected int epochs;
    protected MultiLayerNetwork policy;
    protected Learner dqn;
    public MC_Return(Environment environment, int epochs, MultiLayerNetwork policy, Learner dqn){
        this.environment = environment;
        this.epochs = epochs;
        this.policy = policy;
        this.dqn = dqn;
    }
    public List<Pair<State, double[]>> getReturnMulti(int n){
       List<Pair<State, double[]>> Avg = getReturn();
        for (int i = 0; i < n-1; i++) {
            Avg = getAvg(Avg, getReturn(), (double)(i+2));
        }
        return Avg;
    }
    private List<Pair<State, double[]>> getAvg(List<Pair<State, double[]>> Avg_map, List<Pair<State, double[]>> map, double n){
        int id = 0;
        for (Pair<State, double[]> p : map){
            id = Contains(Avg_map, p.getKey());
            if (id==-1)Avg_map.add(p);
            else{
                Avg_map.set(id, new Pair<>(p.getKey(), getArrayAvg(Avg_map.get(id).getValue(), p.getValue(), n)));
            }
        }
        return Avg_map;
    }
    private int Contains(List<Pair<State, double[]>> data, State item){
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).getKey().equals(item))return i;
        }
        return -1;
    }
    private double[] getArrayAvg(double[] a1, double[] a2, double n){
        double[] ret = new double[a1.length];
        for (int i = 0; i < a1.length; i++) {
            ret[i] = a1[i]+(a2[i]-a1[i])/n;
        }
        return ret;
    }
    private List<Pair<State, double[]>> getReturn(){
        environment.reset();
        Step s;
        List<StateCollector> data = new ArrayList<>();
        while (!environment.isEnd()){
            s = environment.performAction((rnd.nextDouble()<dqn.getMinEpsilon())?rnd.nextInt(dqn.getActionSpaceSize()):dqn.produceActionGreedy(environment.getCurrentState(), policy));
            data = add(s.getR(), s.getA(), s.getBeginState(), data);
        }
        return convert(data);
    }
    private List<Pair<State, double[]>> convert(List<StateCollector> data){
        List<Pair<State, double[]>> ret = new ArrayList<>();
        for(StateCollector stateCollector: data){
            stateCollector.prepare();
            ret.add(new Pair<State, double[]>(stateCollector.getState(), stateCollector.getValue()));
        }
        return ret;
    }
    private List<StateCollector> add(double value, int action, State state, List<StateCollector> data){
        int id = contains(data, state);
        for (StateCollector collector: data){
            collector.update(value);
        }
        if (id==-1){
            StateCollector collector = new StateCollector(state, dqn.getY(), dqn, policy);
            collector.add(value, action);
            data.add(collector);
        }
        return data;
    }
    private int contains(List<StateCollector> data, State item){
        int i = -1;
        for (int j = 0; j < data.size(); j++) {
            if (item.equals(data.get(j).state))return j;
        }
        return i;
    }
}
class StateCollector{
    protected State state;
    protected double[] value;
    protected boolean[] reached;
    protected double y;
    protected Learner dqn;
    protected MultiLayerNetwork net;
    public StateCollector(State state, double y, Learner dqn, MultiLayerNetwork net){
        this.state = state;
        this.y = y;
        this.net = net;
        this.dqn = dqn;
        value = new double[dqn.getActionSpaceSize()];
        reached = new boolean[dqn.getActionSpaceSize()];
        for (int i = 0; i < dqn.getActionSpaceSize(); i++) { reached[i] = false; }
    }
    public void add(double reward, int action){
        reached[action] = true;
         value[action] = value[action]*y+reward*dqn.getRewardScaler();
    }
    public void update(double data){
        for (int i = 0; i < dqn.getActionSpaceSize(); i++) {
            if (!reached[i])continue;
            value[i] = value[i]*y+data;
        }
    }
    public State getState(){
        return state;
    }
    public double[] getValue(){
        return value;
    }
    public void prepare(){
        INDArray input = (dqn.networkType.equals(Learner.InputType.Dense))?Nd4j.create(state.getState()):Nd4j.create(state.getConvVersion());
        double[] output = dqn.computeQ(input, net);
        for (int i = 0; i < dqn.getActionSpaceSize(); i++) {
            if (!reached[i])value[i]=output[i];
        }
    }
}