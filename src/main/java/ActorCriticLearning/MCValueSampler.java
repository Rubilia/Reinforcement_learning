package ActorCriticLearning;

import DQN_learning.Environment.Environment;
import DQN_learning.Learner.Learner;

public class MCValueSampler {
    private Learner learner;
    public MCValueSampler(Learner learner){this.learner = learner; }
    public double evaluate(Environment env, int attemps){
        double value = 0.0;
        for (int i = 0; i < attemps; i++) {
            value+=evaluate(env.clone());
        }
        value/=(double)attemps;
        return value;
    }
    private double evaluate(Environment environment){
        double value = 0.0;
        while (!environment.isEnd()){
            value*=learner.getY();
            value+=environment.performAction(learner.produceActionGreedy(environment.getCurrentState(), learner.getTargetNetwork())).getR();
        }
        return value;
    }
}
