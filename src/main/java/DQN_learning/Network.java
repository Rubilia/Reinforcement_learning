package DQN_learning;

import org.bytedeco.javacv.FrameFilter;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class Network implements Serializable {
    private DQN_Learner.NetworkType type;
    private int actionSpaceSize;
    private MultiLayerNetwork net;
    private MultiLayerNetwork[] nets;
    private boolean useGPU = false;
    private transient ParallelWrapper wrapperOne;
    private transient ParallelWrapper[] wrapperMulti;
    private transient MultiLayerConfiguration config;
    public Network(DQN_Learner.NetworkType type, MultiLayerConfiguration config, int actioSpaceSize, boolean useGPU){
        this.type = type;
        this.useGPU = useGPU;
        this.config = config.clone();
        this.actionSpaceSize = actioSpaceSize;
        if (type.equals(DQN_Learner.NetworkType.OneNetwork)){
            net = new MultiLayerNetwork(config);
            net.init();
            net = net.clone();
            net.setIterationCount(10);
            net.setEpochCount(2);
            if (useGPU)
                wrapperOne = new ParallelWrapper.Builder(net)
                        .prefetchBuffer(64)
                        .workers(4)
                        .averagingFrequency(3)
                        .reportScoreAfterAveraging(false)
                        .build();
            return;
        }
        nets = new MultiLayerNetwork[actioSpaceSize];
        if (useGPU)
           wrapperMulti = new ParallelWrapper[actioSpaceSize];
        for (int i = 0; i < actionSpaceSize; i++) {
            this.nets[i] = (new MultiLayerNetwork(config));
            this.nets[i].init();
            this.nets[i] = this.nets[i].clone();
            this.nets[i].setEpochCount(2);
            this.nets[i].setIterationCount(10);
            if (useGPU)
                this.wrapperMulti[i] = new ParallelWrapper.Builder(nets[i])
                    .prefetchBuffer(24)
                    .workers(4)
                    .averagingFrequency(3)
                    .reportScoreAfterAveraging(false)
                    .build();
        }
    }
    public double[] computeQ(INDArray input){
        double[] Q = new double[actionSpaceSize];
        if (type.equals(DQN_Learner.NetworkType.OneNetwork)){
            INDArray out = net.output(input);
            for (int i = 0; i < actionSpaceSize; i++) {
                Q[i] = out.getDouble(i);
            }
        }
        else{
            for (int i = 0; i < actionSpaceSize; i++) {
                Q[i] = nets[i].output(input).getDouble(0);
            }
        }
        return Q;
    }
    public void LearnExperienceReplay(Step[] data, DQN_Learner dqn){
        if (type.equals(DQN_Learner.NetworkType.OneNetwork)) LearnOne(data, dqn);
        else LearnMulti(data, dqn);
    }
    private void LearnOne(Step[] data, DQN_Learner dqn){
        QLearningDataFetcher fetcher = new QLearningDataFetcher(1, dqn.networkType);
        double[] Q;
        for (int i = 0; i < data.length; i++) {
            Q = dqn.getTargetNetwork().computeQ(data[i].getBeginState(dqn.networkType));
            if (data[i].isTerminate()){Q[data[i].getA()] = data[i].getR()*dqn.getRewardScaler();}
            else{ Q[data[i].getA()] = data[i].getR()*dqn.getRewardScaler()+dqn.getY()*dqn.getMaxQ(dqn.getPastNetwork().computeQ(data[i].getEndState(dqn.networkType))); }
            fetcher.addNewValueConv(data[i].getBeginStateConv(), Q);
        }
        QLearningDataSetIterator dataSetIterator;
        dataSetIterator = new QLearningDataSetIterator(Math.min(dqn.getBatchLearningSize(), data.length), data.length, fetcher);
        if (!useGPU) { for (int i = 0; i < dqn.getLearningEpochsPerIteration(); i++) { net.fit(fetcher.getDataSet());} }
        else{
            for (int i = 0; i < dqn.getLearningEpochsPerIteration(); i++) {
                wrapperOne.fit(dataSetIterator);
            }
        }
    }
    private void LearnMulti(Step[] data, DQN_Learner dqn){
        Map<Integer, QLearningDataFetcher> fetcher = new HashMap<>();
        double[] Q;
        for (int i = 0; i < data.length; i++) {
            Q = dqn.getTargetNetwork().computeQ(data[i].getBeginState(dqn.networkType));
            if (data[i].isTerminate()){Q[data[i].getA()] = data[i].getR()*dqn.getRewardScaler();}
            else{ Q[data[i].getA()] = data[i].getR()*dqn.getRewardScaler()+dqn.getY()*dqn.getMaxQ(dqn.getPastNetwork().computeQ(data[i].getEndState(dqn.networkType))); }
            if (fetcher.containsKey(data[i].getA())){
                fetcher.get(data[i].getA()).addNewValueConv(data[i].getBeginStateConv(), new double[] {Q[data[i].getA()]}); }
            else{
                fetcher.put(data[i].getA(), new QLearningDataFetcher(1, dqn.networkType));
                fetcher.get(data[i].getA()).addNewValueConv(data[i].getBeginStateConv(), new double[] {Q[data[i].getA()]}); }
        }
        QLearningDataSetIterator dataSetIterator;
        for (int action: fetcher.keySet()){
            dataSetIterator = new QLearningDataSetIterator(Math.min(dqn.getBatchLearningSize(), data.length), data.length, fetcher.get(action));
            if (!useGPU) { for (int i = 0; i < dqn.getLearningEpochsPerIteration(); i++) { nets[action].fit(fetcher.get(action).getDataSet());} }
            else{
                for (int i = 0; i < dqn.getLearningEpochsPerIteration(); i++) {
                    wrapperMulti[action].fit(dataSetIterator);
                }
            }
        }
    }
    public MultiLayerNetwork[] getNetwork(){
        if (type.equals(DQN_Learner.NetworkType.OneNetwork))
            return new MultiLayerNetwork[]{net};
        return nets;
    }
    public void save(String path) {
        try(ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path)))
        {
            oos.writeObject(this);
        }
        catch(Exception ex){ }
    }
    public Network read(String path){
        Network net = null;
        try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path)))
        {
            net = (Network) ois.readObject();
        }
        catch(Exception ex){ }
        return net;
    }
    @Override
    protected Network clone() {
        return new Network(this.type, config, actionSpaceSize, useGPU);
    }
}
