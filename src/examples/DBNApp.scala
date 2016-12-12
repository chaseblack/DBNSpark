package examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import breeze.linalg.DenseMatrix
import breeze.linalg.{DenseMatrix => BDM}
import DBN.DBN
import neuralnet.NeuralNet
import breeze.linalg.{DenseMatrix => BDM}

object DBNApp {
    def extract(samples:RDD[String])={
    val a=samples.first().split(",").size-1//why "num_vislayer-1"? cause the first column is label which are excluded
    val b=samples.map{
     line =>
     val f1 = line.split(",")
     val f =f1.map(f =>f.toDouble)
     val y = Array(f(0))
     val x =f.slice(1,f.length)
     /***********************ATTENTION**************************************
      * Mathematically, each sample in the training set should be a DenseVector. 
      * However, in this code each sample is wrapped in a DenseMatrix. Why? I personally
      * think this is because multiplication between a vector and a matrix in Breeze is tricky
      * from my prior experience on Breeze.
      * 
      * BDM(1,x.length,x) is actually a vector, mathematically. the row number is 1, 
      * the col number is x.length, and the vector data are from x itself. 
      * Therefore I denote those BDMs that are actually vectors.
     **/
     (new BDM(1,y.length,y),new BDM(1,x.length,x))
    }
    (a,b)
  }
  
  def main(args:Array[String]){
    val file="data.csv"
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
    val sc = new SparkContext(conf)
    val samples = sc.textFile(file, 4).cache()
    val tuple=DBNApp.extract(samples) 
    val num_vislayer=tuple._1
    val train_d =tuple._2
    val opts = Array(100.0,20.0,5)//batchsize, numepoch, k_CD 
    val DBNmodel =new DBN().setSize(Array(num_vislayer, 7, 1)).setLayer(3).setMomentum(0.1).setAlpha(0.2).DBNtrain(train_d, opts)
    
    val mynn = DBNmodel.dbnunfoldtonn(1)
    val nnopts = Array(100.0, 50.0, 0.0)
    val numExamples = train_d.count()
    println(s"numExamples = $numExamples.")
    println(mynn._2)
    for (i <- 0 to mynn._1.length - 1) {
      print(mynn._1(i) + "\t")
    }
    println()
    println("mynn_W1")
    val tmpw1 = mynn._3(0)
    for (i <- 0 to tmpw1.rows - 1) {
      for (j <- 0 to tmpw1.cols - 1) {
        print(tmpw1(i, j) + "\t")
      }
      println()
    }
    val NNmodel = new NeuralNet().setSize(mynn._1).setLayer(mynn._2).setActivation_function("sigm").setOutput_function("sigm").setInitW(mynn._3).NNtrain(train_d, nnopts)
    //5 NN模型测试
    val NNforecast = NNmodel.predict(train_d)
    val NNerror = NNmodel.Loss(NNforecast)
    println(s"NNerror = $NNerror.")
    val printf1 = NNforecast.map(f => (f.label.data(0), f.predict_label.data(0))).take(200)
    println("预测结果——实际值：预测值：误差")
    for (i <- 0 until printf1.length)
      println(printf1(i)._1 + "\t" + printf1(i)._2 + "\t" + (printf1(i)._2 - printf1(i)._1))
  }
}