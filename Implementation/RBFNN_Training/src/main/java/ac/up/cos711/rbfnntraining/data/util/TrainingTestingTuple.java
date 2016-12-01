package ac.up.cos711.rbfnntraining.data.util;

import ac.up.cos711.rbfnntraining.data.Dataset;

/**
 *
 * @author Abrie van Aardt
 */
public class TrainingTestingTuple {
    
    public TrainingTestingTuple(Dataset _training, Dataset _testing){
        training = _training;
        testing = _testing;
    }
    
    public Dataset training;
    public Dataset testing;
}
