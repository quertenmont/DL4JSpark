package LQ

//SCALA
import scala.reflect.ClassTag
import scala.collection.mutable.ArrayBuffer

//SPARK
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Dataset
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.rdd._
import org.apache.spark.util._

//JAVA MISC
import java.util.regex.Matcher
import java.util.regex.Pattern
import java.io.PrintWriter


//DL4J
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.api.{Layer, OptimizationAlgorithm}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.EventStats;
import org.deeplearning4j.spark.stats.StatsUtils;


object LQTests {

  def main(args: Array[String]) {
    if (args.length > 1) {
      args.foreach{ println }
    }

 
    val spark = SparkSession
       .builder()
       .master("local[2]")
       .appName("DL4J Testing")
       .config("spark.driver.memory", "4g")
       .config("spark.executor.memory", "2g")
       .config("spark.driver.extraJavaOptions", "-Dorg.bytedeco.javacpp.maxbytes=104857600") //100MB
       .config("spark.executor.extraJavaOptions", "-Dorg.bytedeco.javacpp.maxbytes=104857600") //100MB
       .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val sparkContext = spark.sparkContext  

  val dsList = Range(0,100).map{ _ =>     
    val input  = Nd4j.zeros(4, 2) //4x2 array
    val labels = Nd4j.zeros(4, 2)
  
    var index = 0
    for(  bit1 <- 0 to 1 ){
      for(  bit2 <- 0 to 1 ){
         input .putScalar(Array[Int](index, 0), bit1)
         input .putScalar(Array[Int](index, 1), bit2)
         labels.putScalar(Array[Int](index, 0), if( (bit1==1)^(bit2==1) ) 1 else 0 )
         labels.putScalar(Array[Int](index, 1), if( (bit1==1)^(bit2==1) ) 0 else 1 )
         index = index+1
      }
    }
  
    // create dataset object
    val ds = new DataSet(input, labels);
    ds
  }
  
  
  val trainingData = sparkContext.parallelize(dsList)
  val conf = new NeuralNetConfiguration.Builder()
  .iterations(200)
  .learningRate(0.1)
  .seed(123)
  .useDropConnect(false)
  .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
  .updater(Updater.NESTEROVS).momentum(0.9)
  .biasInit(0)
  .miniBatch(false)
  .list()
  //add the layers
  .layer(0,  new DenseLayer.Builder()
     .nIn(2)
     .nOut(4)
     .activation("sigmoid")
     .weightInit(WeightInit.XAVIER)
     .build())
  .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
    .nIn(4)
    .nOut(2)
    .activation("softmax")
    .weightInit(WeightInit.XAVIER)
    .build())
  .pretrain(false)
  .backprop(true)
  .build();

  val tm = new ParameterAveragingTrainingMaster.Builder(1)
     .workerPrefetchNumBatches(2)    //Async prefetch 2 batches for each worker
     .averagingFrequency(1)
     .batchSizePerWorker(1)
     .build();

  val sparkNet = new SparkDl4jMultiLayer(sparkContext, conf, tm);
  sparkNet.setCollectTrainingStats(true);

  sparkNet.fit(trainingData);  //only one epoch for now

  ///FIXME don't evaluate for now on
  //Perform evaluation (distributed)
  //val evaluation = sparkNet.evaluate(testData);
  //println("***** Evaluation *****");
  //println(evaluation.stats());

  //Delete the temp training files, now that we are done with them
  tm.deleteTempFiles(sparkContext);
  }
}


