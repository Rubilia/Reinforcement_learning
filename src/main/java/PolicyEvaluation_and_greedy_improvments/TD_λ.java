package PolicyEvaluation_and_greedy_improvments;

import Tools.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static Tools.OtherTools.cut;
import static Tools.OtherTools.create;

public class TD_λ {
    static List<Double> values = new ArrayList<>(), realValues = new ArrayList<>();
    static double zeroRevard = -1.0, endReward = 1.0;
    static long iterationsAmount = 1000, epochsAmount = 2;
    static int iterations = 0, statesAmount = 5;
    static double alpha, y = 0.9, λ = 0.8;
    static int i = 0;
    public static void main(String[] args) {
        alpha=5/(((double)iterationsAmount)/(double)statesAmount);
        System.out.println("Below written values for V(s) for Random Walk. Each learning iteration is highlighted by #n===, n - iteration id");
        values= create(statesAmount+2);
        realValues = create(statesAmount+2);
        System.out.println("#1=============================================");
        for (i = 0; i < epochsAmount; i++) {
            evaluate.run();
            realValues.clear();
            realValues.addAll(values);
            realValues.set(0, realValues.get(1));
            realValues.set(statesAmount+1, realValues.get(statesAmount));
            values = create(statesAmount+2);
            iterations=0;
            if (i==0||i==epochsAmount-1){Tools.Graph.buildGraphRW_Vs(cut(realValues), (i+1));}
            if (i < epochsAmount-1) System.out.println("#"+(i+2)+"=============================================");
        }
    }
    static Runnable evaluate = new Runnable() {
        //Random walk
        Random rnd = new Random();
        int currentPos;
        List<Double> E;
        void init(){
            currentPos = rnd.nextInt(statesAmount);
            E = create(statesAmount+2);
        }
        @Override
        public void run() {
            init();
            while (iterations<iterationsAmount){
                while (currentPos!=-1&&currentPos!=statesAmount){
                    update(move());
                }
                init();
                iterations++;
            }
        }

        Pair<Double, Integer> move(){
            int ret = currentPos;
            boolean left;
            if (realValues.get(currentPos)-realValues.get(currentPos+2)==0.0) {
                left=rnd.nextBoolean();
            }
            else {
                left = realValues.get(currentPos)>realValues.get(currentPos+2);
            }
            if (left){
                currentPos--;
                if (currentPos==-1){return new Pair<>(zeroRevard, ret);}
                else {return new Pair<>(0.0, ret);}
            }
            else{
                currentPos++;
                if (currentPos==statesAmount){return new Pair<>(endReward, ret);}
                else {return new Pair<>(0.0, ret);}
            }
        }

        public void update(Pair<Double, Integer> reward){
            E.set(reward.getValue(), E.get(reward.getValue())*y*λ+1);
            double TD_error = reward.getKey()+y*values.get(currentPos+1)-values.get(reward.getValue()+1);
            for (int j = 0; j < statesAmount; j++) {
                values.set(j+1, values.get(j+1)+alpha*TD_error*E.get(j));
            }
        }
    };

}
