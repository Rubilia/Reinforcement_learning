package Tools;

import java.util.ArrayList;
import java.util.List;

public class OtherTools {
    public static List<Double> cut(List<Double> input){
        List<Double> ret = new ArrayList<>();
        for (int j = 1; j < input.size()-1; j++) {
            ret.add(input.get(j));
        }
        return ret;
    }
    public static List<Double> create(int n){
        List<Double> ret = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            ret.add(0.0);
        }
        return ret;
    }

}
