package ac.up.cos711.rbfnntraining.experiment;

import ac.up.cos711.rbfnntraining.data.Dataset;
import ac.up.cos711.rbfnntraining.data.Graph;
import ac.up.cos711.rbfnntraining.data.Results;
import ac.up.cos711.rbfnntraining.function.problem.NetworkError;
import ac.up.cos711.rbfnntraining.function.problem.RealProblem;
import ac.up.cos711.rbfnntraining.neuralnet.RBFNeuralNetTest;
import ac.up.cos711.rbfnntraining.neuralnet.metric.ClassificationAccuracy;
import ac.up.cos711.rbfnntraining.neuralnet.metric.DefaultNetworkError;
import ac.up.cos711.rbfnntraining.neuralnet.training.GradientDescent;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Class representing an abstract experiment that provides hooks for specific
 * experiments to perform their desired tasks.
 *
 * @author Abrie van Aardt
 */
public class GDExperiment extends Thread {

    /**
     *
     * @param _numSim
     * @param _maxEpoch
     * @param _numCentres
     * @param _acceptableTError
     * @param w_learningRate
     * @param u_learningRate
     * @param sigma_learningRate
     * @param d_max
     * @param _classificationRigor
     */
    public GDExperiment(int _numSim, int _maxEpoch, int _numCentres, double _acceptableTError, double w_learningRate, double u_learningRate, double sigma_learningRate, double d_max, double _classificationRigor) {
        name = "GD_Experiment_" + _numCentres;
        path = "Experiment/" + name;
        numSim = _numSim;
        simulations = new double[numSim];
        epochs = new double[_maxEpoch];
        numCentres = _numCentres;

        //note the dataPartitions are recorded in alphabetical order, easy to remember this way
        datasetNames = new String[]{
            "Cancer",
            "Diabetes",
            "Glass",
            "Heart",
            "Iris"
        };

        NUM_DATASETS = datasetNames.length;

        avgTrainingErrorHistories = new double[NUM_DATASETS][_maxEpoch];
        avgValidationErrorHistories = new double[NUM_DATASETS][_maxEpoch];

        trainingErrors = new double[numSim + 2][NUM_DATASETS];
        generalErrors = new double[numSim + 2][NUM_DATASETS];
        accuracies = new double[numSim + 2][NUM_DATASETS];

        avgErrorIndex = numSim;
        stdDevErrorIndex = avgErrorIndex + 1;

        for (int i = 0; i < simulations.length; i++) {
            simulations[i] = i + 1;
        }

        for (int i = 0; i < epochs.length; i++) {
            epochs[i] = i + 1;
        }

        ACCEPTABLE_TRAINING_ERROR = _acceptableTError;
        w_LEARNING_RATE = w_learningRate;
        u_LEARNING_RATE = u_learningRate;
        sigma_LEARNING_RATE = sigma_learningRate;
        d_MAX = d_max;
        MAX_EPOCH = _maxEpoch;
        CLASSIFICATION_RIGOR = _classificationRigor;
    }

    @Override
    public void run() {

        Logger
                .getLogger(GDExperiment.class.getName())
                .log(Level.INFO, "...");

        Logger
                .getLogger(GDExperiment.class.getName())
                .log(Level.INFO, "Running experiment: {0}", name);

        Logger
                .getLogger(GDExperiment.class.getName())
                .log(Level.INFO, "Doing {0} simulation(s)", numSim);

        try {

            for (int i = 1; i <= numSim; i++) {

                Logger
                        .getLogger(GDExperiment.class.getName())
                        .log(Level.INFO, name + " - Simulation {0}", i);

                runSimulation(i);

            }

            finalise();

        }
        catch (Exception e) {
            Logger.getLogger(GDExperiment.class.getName()).log(Level.SEVERE, "", e);
        }
    }

    /**
     * Calculates the running average assuming indexing starts at
     *
     * @param oldAvg
     * @param newValue
     * @param currentNumber
     * @return new average
     */
    protected double calculateNewAverage(double oldAvg, double newValue, int currentNumber) {
        return (oldAvg * (currentNumber - 1) + newValue) / currentNumber;
    }

    /**
     * Calculates the sample standard deviation of the array of values
     *
     * @param values
     * @param avg
     * @return sample standard deviation
     */
    protected double calculateSampleStdDev(double[] values, double avg) {
        double stdDev = Arrays.stream(values).map(i -> Math.pow(i - avg, 2)).sum() / (values.length - 1);
        return Math.sqrt(stdDev);
    }

    protected void runSimulation(int currentSimulation)
            throws Exception {

        for (int i = 0; i < NUM_DATASETS; i++) {
            Dataset dataset = Dataset
                    .fromFile(PATH_PREFIX
                            + "/" + datasetNames[i].toLowerCase()
                            + EXT);
            dataset.setDatasetName(datasetNames[i]);

            List<Dataset> dataPartitions = dataset.split(0.6, 0.3, 0.1);
            RBFNeuralNetTest network = new RBFNeuralNetTest(dataset.getInputCount(), numCentres, dataset.getTargetCount());
            GradientDescent gd = new GradientDescent(ACCEPTABLE_TRAINING_ERROR, w_LEARNING_RATE, u_LEARNING_RATE, sigma_LEARNING_RATE, d_MAX, CLASSIFICATION_RIGOR, MAX_EPOCH);

            gd.train(network, dataPartitions.get(0), dataPartitions.get(1));

            trainingErrors[currentSimulation - 1][i] = gd.getTrainingError();

            double tempGeneralError = new DefaultNetworkError().measure(network, dataPartitions.get(2));
            double tempAccuracy = new ClassificationAccuracy(CLASSIFICATION_RIGOR).measure(network, dataPartitions.get(2));
            generalErrors[currentSimulation - 1][i] = tempGeneralError;
            accuracies[currentSimulation - 1][i] = tempAccuracy;

            trainingErrors[avgErrorIndex][i] = calculateNewAverage(trainingErrors[avgErrorIndex][i], gd.getTrainingError(), currentSimulation);
            generalErrors[avgErrorIndex][i] = calculateNewAverage(generalErrors[avgErrorIndex][i], tempGeneralError, currentSimulation);
            accuracies[avgErrorIndex][i] = calculateNewAverage(accuracies[avgErrorIndex][i], tempAccuracy, currentSimulation);

            //calculate standard deviation when finalising
            double[] tempTrainingHist = gd.getTrainingErrorHistory();
            double[] tempValidationHist = gd.getValidationErrorHistory();

            //update average error histories (errors over epochs)
            for (int j = 0; j < MAX_EPOCH; j++) {
                //dataset i epoch j
                avgTrainingErrorHistories[i][j] = calculateNewAverage(avgTrainingErrorHistories[i][j], tempTrainingHist[j], currentSimulation);
                avgValidationErrorHistories[i][j] = calculateNewAverage(avgValidationErrorHistories[i][j], tempValidationHist[j], currentSimulation);
            }
        }

    }

    protected void finalise() throws Exception {

        //calculate std dev for each type of measure
        for (int i = 0; i < NUM_DATASETS; i++) {

            //column of errors
            double[] errorColumn = new double[numSim];

            //extract column of trainingError
            for (int j = 0; j < numSim; j++) {
                errorColumn[j] = trainingErrors[j][i];
            }
            trainingErrors[stdDevErrorIndex][i] = calculateSampleStdDev(errorColumn, trainingErrors[avgErrorIndex][i]);

            //extract column of generalisationError
            for (int j = 0; j < numSim; j++) {
                errorColumn[j] = generalErrors[j][i];
            }
            generalErrors[stdDevErrorIndex][i] = calculateSampleStdDev(errorColumn, generalErrors[avgErrorIndex][i]);

            //extract column of accuracy
            for (int j = 0; j < numSim; j++) {
                errorColumn[j] = accuracies[j][i];
            }
            accuracies[stdDevErrorIndex][i] = calculateSampleStdDev(errorColumn, accuracies[avgErrorIndex][i]);
        }
        
        //raw result output
        Results.writeToFile(path, name + "_TrainingErrors", trainingErrors);
        Results.writeToFile(path, name + "_GeneralisationError", generalErrors);
        Results.writeToFile(path, name + "_Accuracies", accuracies);

        //graphing the results
        //add a plot for each of the datasets (for now)
        Results.newGraph(this, path + "/Plots", "Error vs Epoch", "Epoch", "Error", "", 2);
        for (int i = 0; i < NUM_DATASETS; i++) {
            Results.addPlot(this, datasetNames[i] + " E_T", epochs, avgTrainingErrorHistories[i], "line lt " + (i + 1));
            Results.addPlot(this, datasetNames[i] + " E_V", epochs, avgValidationErrorHistories[i], "line dt 2 lt " + (i + 1));
        }
        Results.plot(this);

        
    }

    public String getExpName() {
        return name;
    }

    public Graph graph;
    protected String name;
    protected String path;
    protected RealProblem problem;
    protected double[] simulations;
    protected double[] epochs;
    protected int stdDevErrorIndex;
    protected int avgErrorIndex;
    private int numSim;

    private final double[][] avgTrainingErrorHistories;
    private final double[][] avgValidationErrorHistories;

    private final double[][] trainingErrors;
    private final double[][] generalErrors;
    private final double[][] accuracies;

    private final double ACCEPTABLE_TRAINING_ERROR;
    private final double w_LEARNING_RATE;
    private final double u_LEARNING_RATE;
    private final double sigma_LEARNING_RATE;
    private final double d_MAX;
    private final int MAX_EPOCH;
    private final double CLASSIFICATION_RIGOR;

    private final int NUM_DATASETS;

    //RBF params
    private int numCentres;

    private final String PATH_PREFIX = "ac/up/cos711/rbfnntraining/data/";
    private final String EXT = ".nsds";

    private final String[] datasetNames;
}
