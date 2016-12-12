package examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import scala.collection.mutable.ArrayBuffer
import RBM.RBM
import breeze.linalg.InjectNumericOps

object MyApp {
   def main(args:Array[String]){
     
      //val file = "/home/chase/Documents/0my_projects/WorkloadsPrediction4GoogleCloud/clusterdatatrain/cpu/predict-raw/raw-in5-i.csv"
     val file="raw-in10-i.csv"
      /*
      /****gen rand matrix 4 training****/
      scala.tools.nsc.io.File("Data.txt").writeAll(DenseMatrix.rand(RBM.n_visible,100).toString)
      */
      
      val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
      val sc = new SparkContext(conf)
      val Data = sc.textFile(file, 4).cache()
      
      val errArray=new ArrayBuffer[Double]()
      
      val n_epoch=1000;
      for(i<- 1 to n_epoch){
        val vP_map=Data.map(RBM.tune)
        val vP_sum=vP_map.reduce((vP1,vP2)=>(vP1._1 + vP2._1, vP1._2 + vP2._2, vP1._3 + vP2._3, vP1._4 + vP2._4))
        RBM.vW=RBM.momentum*RBM.vW+vP_sum._1/(1.0*vP_map.count())
        RBM.vb=RBM.momentum*RBM.vb+vP_sum._2/(1.0*vP_map.count())
        RBM.vc=RBM.momentum*RBM.vc+vP_sum._3/(1.0*vP_map.count())
        println(f"the $i epoch. Error:"+vP_sum._4)
        errArray+=vP_sum._4
        RBM.W=RBM.W+RBM.vW
        RBM.b=RBM.b+RBM.vb
        RBM.c=RBM.c+RBM.vc
        }
      //val data = List("everything", "you", "want", "to", "write", "to", "the", "file")
      import java.io._
      val error_file = "error.txt"
      val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(error_file)))
      for (x <- errArray) {
       writer.write(x + ",")  // however you want to format it
      }
      writer.close()
      sc.stop()
    }
}