package ActorCriticLearning;

import DQN_learning.Environment.Environment;
import DQN_learning.Environment.State;
import DQN_learning.Environment.Step;
import DQN_learning.Learner.Learner;
import Tools.Pair;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

public class A2C_Learner extends Learner {
    private Learner.InputType policyType, valueType;
    private Random rnd = new Random();
    private int actionSpaceSize;
    private int MCAttemps;
    private int scoreListener = 100;
    private int LearningEpochsPerIterationValue = 16;
    private Environment environment;
    private MultiLayerNetwork policy, value;
    private double alpha = 0.05;
    private double y = 0.95;
    private boolean MCAdvantage;
    private double rewardScaler = 1.0;
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }
    public A2C_Learner(MultiLayerConfiguration policy, MultiLayerConfiguration value, InputType policyType, InputType valueType) {
        this.MCAdvantage = false;
        this.MCAttemps = 0;
        this.policy = new MultiLayerNetwork(policy);
        this.policy.init();
        this.value = new MultiLayerNetwork(value);
        this.value.init();
        this.policyType = policyType;
        this.valueType = valueType;
        try { (new File("Log.txt")).delete(); Files.createFile(Paths.get("Log.txt")); Files.write(Paths.get("Log.txt"), "".getBytes(), StandardOpenOption.WRITE); } catch (IOException e) { }
    }
    public A2C_Learner(MultiLayerConfiguration policy, InputType policyType, int AttempsPerEvaluation){
        this.MCAdvantage = true;
        this.MCAttemps = AttempsPerEvaluation;
        this.policy = new MultiLayerNetwork(policy);
        this.policy.init();
        this.policyType = policyType;
    }
   public void setLearningEpochsPerIterationValue(int learningEpochsPerIterationValue) {
        LearningEpochsPerIterationValue = learningEpochsPerIterationValue;
    }
    public Environment getEnvironment() {
        return environment;
    }
    public void setEnvironment(Environment environment) {
        this.environment = environment;
    }
    public double getY() {
        return y;
    }
    public void setY(double y) {
        this.y = y;
    }
    @Override
    public MultiLayerNetwork getTargetNetwork() {
        return policy;
    }
    public void setRewardScaler(double rewardScaler) {
        this.rewardScaler = rewardScaler;
    }
    public void setScoreListener(int scoreListener) {
        this.scoreListener = scoreListener;
    }
    public void setActionSpaceSize(int actionSpaceSize) {
        this.actionSpaceSize = actionSpaceSize;
    }
    public void Learn(int epochs){
        environment.reset();
        for (int i = 0; i < epochs; i++) {
            environment.reset();
            Step s;
            while (!environment.isEnd()){
                Environment last = environment.clone();
                s = environment.performAction(produceAction(environment.getCurrentState(), policy));
                learnPolicy(s, last);
                learnValue(s);
            }
            if (i%scoreListener==0) {
                Pair<String, Boolean> score = (environment).getScore(this);
                Log("epoch #" + i + ", " + score.getKey());
                System.out.println("##############################################################");
                System.out.println("epoch #" + i + ", " + score.getKey());
                System.out.println("##############################################################");
                if (score.getValue()) { return; }
            }
        }
        environment.reset();
    }
    public void Log(String txt){
        txt+="\n";
        try {
            Files.write(Paths.get("Log.txt"), txt.getBytes(), StandardOpenOption.APPEND);
        }catch (Exception e){}
    }
    private void learnPolicy(Step s, Environment last){
        MultiLayerNetwork policyFrozen = policy.clone();
        policyFrozen.setInput(s.getBeginState(policyType));
        policyFrozen.feedForward(true, false);
        INDArray errs = Transforms.pow(policy.output(s.getEndState(policyType)), -1);
        Gradient gradientLog = policyFrozen.backpropGradient(errs, null).getFirst();
        double advantage = computeAdvantage(s, value, last);
        INDArray newParams = gradientLog.gradient().muli(advantage*alpha);
        policy.params().add(newParams);
    }
    private void learnValue(Step s){
        if (MCAdvantage)return;
        List<Step> data = sampleData(s);
        double[][] ValueTarget = new double[data.size()][1];
        State[] input = new State[data.size()];
        for (int i = 0; i < data.size(); i++) {
            input[i] = data.get(i).getBeginState();
            ValueTarget[i][0] = data.get(i).getR() * rewardScaler + y*compute(data.get(i).getEndState(), value, 1, valueType)[0];
        }
        DataSet dataSet;
        if (valueType.equals(Learner.InputType.Dense)){
            double[][] inputs = new double[data.size()][];
            for (int i = 0; i < data.size(); i++) {
                inputs[i] = data.get(i).getBeginStateDense();
            }
            dataSet = new DataSet(Nd4j.create(inputs), Nd4j.create(ValueTarget));
        }
        else{
            double[][][][] inputs = new double[data.size()][][][];
            for (int i = 0; i < data.size(); i++) {
                inputs[i] = data.get(i).getBeginStateConv();
            }
            dataSet = new DataSet(Nd4j.create(inputs), Nd4j.create(ValueTarget));
        }
        for (int i = 0; i < LearningEpochsPerIterationValue; i++) {
            value.fit(dataSet);
        }
    }
    private List<Step> sampleData(Step s){
        List<Step> ret = new ArrayList<>();
        Environment env = environment.clone();
        env.reset();
        while (env.isEnd()){
            ret.add(env.performAction(produceActionGreedy(env.getCurrentState(), policy)));
        }
        ret.add(s);
        return ret;
    }
    private double computeAdvantage(Step s, MultiLayerNetwork value, Environment lastEnv){
        double advantage;
        if (MCAdvantage){
            MCValueSampler valueSampler = new MCValueSampler(this);
            double Q = s.getR()+valueSampler.evaluate(environment.clone(), MCAttemps), V = valueSampler.evaluate(lastEnv, MCAttemps);
            advantage = Q-V;
        }
        else{
            advantage = s.getR() * rewardScaler + y*compute(s.getEndState(), value, 1, valueType)[0] - compute(s.getBeginState(), value, 1, valueType)[0];
        }
        return advantage;
    }
    public int produceAction(State s, MultiLayerNetwork net){
        double[] output = compute(s, net, actionSpaceSize, policyType);
        DistributedRandomNumberGenerator dng = new DistributedRandomNumberGenerator();
        for (int i = 0; i < output.length; i++) {
            dng.addNumber(i, Math.exp(output[i]));
        }
        return dng.getDistributedRandomNumber();
    }

    @Override
    public int produceActionGreedy(State s, MultiLayerNetwork net) {
        return produceAction(s, net);
    }

    public double[] compute(State input, MultiLayerNetwork net, int outputSize, InputType type){
        if (type.equals(Learner.InputType.Dense))return compute(Nd4j.create(input.getState()), net, outputSize);
        else return compute(Nd4j.create(input.getConvVersion()), net, outputSize);
    }
    public double[] compute(INDArray input, MultiLayerNetwork net, int outputSize){
        double[] Output = new double[outputSize];
        INDArray out = net.output(input);
        for (int i = 0; i < outputSize; i++) {
            Output[i] = out.getDouble(i);
        }
        return Output;
    }
    public MultiLayerNetwork getPolicy(){ return policy; }
    public MultiLayerNetwork getValue(){ return value; }
}
class DistributedRandomNumberGenerator {

    private Map<Integer, Double> distribution;
    private double distSum;
    public DistributedRandomNumberGenerator() {
        distribution = new HashMap<>();
    }
    public void addNumber(int value, double distribution) {
        if (this.distribution.get(value) != null) {
            distSum -= this.distribution.get(value);
        }
        this.distribution.put(value, distribution);
        distSum += distribution;
    }

    public int getDistributedRandomNumber() {
        double rand = Math.random();
        double ratio = 1.0f / distSum;
        double tempDist = 0;
        for (Integer i : distribution.keySet()) {
            tempDist += distribution.get(i);
            if (rand / ratio <= tempDist) {
                return i;
            }
        }
        return 0;
    }

}