package DBN

import breeze.stats.distributions.Gaussian
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets

object DBNResult {
  def show()={
    /*
    println()
    val r1=DenseVector.rand(200,new Gaussian(0.6,0.1))
    val rand=new scala.util.Random()
    val r3=r1.map { x => x+ rand.nextGaussian()*0.04}
    Files.write(Paths.get("file1.txt"), r1.toString().getBytes(StandardCharsets.UTF_8))
    Files.write(Paths.get("file3.txt"), r3.toString().getBytes(StandardCharsets.UTF_8))
    */
    Runtime.getRuntime().exec("./show.sh");
    //ps.waitFor();
  }
}