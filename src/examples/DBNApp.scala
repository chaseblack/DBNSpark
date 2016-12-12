package examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import breeze.linalg.DenseMatrix
import breeze.linalg.{DenseMatrix => BDM}
import DBN.DBN
import breeze.linalg.{DenseMatrix => BDM}

object DBNApp {
  def main(args:Array[String]){
    
     val file="data.csv"
     val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
     val sc = new SparkContext(conf)
     val examples = sc.textFile(file, 4).cache()
     
     val train_d =examples.map { line =>
     val f1 = line.split(",")
     val f =f1.map(f =>f.toDouble)
     val y = Array(f(1))
     val x =f.slice(2,f.length)
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
    
    val opts = Array(100.0,20.0,0.0) 
    //3 设置训练参数，建立DBN模型
    val DBNmodel =new DBN().setSize(Array(5, 7)).setLayer(2).setMomentum(0.1).setAlpha(0.2).DBNtrain(train_d, opts) 
    
    //DBNResult.show()
  }
}