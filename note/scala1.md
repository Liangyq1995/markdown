Scala 使用 package 关键字定义包，在Scala将代码定义到某个包中有两种方式：

第一种方法和 Java 一样，在文件的头定义包名，这种方法就后续所有代码都放在该包中。

```scala
package com.runoob

```

Scala 使用 import 关键字引用包。

```scala
import java.awt.Color  // 引入Color
 
import java.awt._  // 引入包内所有成员
 
def handler(evt: event.ActionEvent) { // java.awt.event.ActionEvent
  ...  // 因为引入了java.awt，所以可以省去前面的部分
}
import java.awt.{Color, Font}
 
// 重命名成员
import java.util.{HashMap => JavaHashMap}
 
// 隐藏成员
import java.util.{HashMap => _, _} // 引入了util包的所有成员，但是HashMap被隐藏了
```

> 默认情况下，Scala 总会引入 java.lang._ 、 scala._ 和 Predef._，这里也能解释，为什么以scala开头的包，在使用时都是省去scala.的。

###### 数据类型

| 数据类型 | 描述                                                         |
| :------- | :----------------------------------------------------------- |
| Byte     | 8位有符号补码整数。数值区间为 -128 到 127                    |
| Short    | 16位有符号补码整数。数值区间为 -32768 到 32767               |
| Int      | 32位有符号补码整数。数值区间为 -2147483648 到 2147483647     |
| Long     | 64位有符号补码整数。数值区间为 -9223372036854775808 到 9223372036854775807 |
| Float    | 32 位, IEEE 754 标准的单精度浮点数                           |
| Double   | 64 位 IEEE 754 标准的双精度浮点数                            |
| Char     | 16位无符号Unicode字符, 区间值为 U+0000 到 U+FFFF             |
| String   | 字符序列                                                     |
| Boolean  | true或false                                                  |
| Unit     | 表示无值，和其他语言中void等同。用作不返回任何结果的方法的结果类型。Unit只有一个实例值，写成()。 |
| Null     | null 或空引用                                                |
| Nothing  | Nothing类型在Scala的类层级的最底端；它是任何其他类型的子类型。 |
| Any      | Any是所有其他类的超类                                        |
| AnyRef   | AnyRef类是Scala里所有引用类(reference class)的基类           |

整型字面量用于 Int 类型，如果表示 Long，可以在数字后面添加 L 或者小写 l 作为后缀。如果浮点数后面有f或者F后缀时，表示这是一个Float类型，否则就是一个Double类型的。

在 Scala 字符变量使用单引号 **'** 来定义

在 Scala 字符串字面量使用双引号 **"** 来定义

多行字符串用三个双引号来表示分隔符，格式为：**""" ... """**。

###### 变量

- 一、变量： 在程序运行过程中其值可能发生改变的量叫做变量。如：时间，年龄。
- 二、常量 在程序运行过程中其值不会发生变化的量叫做常量。如：数值 3，字符'A'。

在 Scala 中，使用关键词 **"var"** 声明变量，使用关键词 **"val"** 声明常量。

```scala
var VariableName : DataType [=  Initial Value]
val VariableName : DataType [=  Initial Value]

var myVar : String = "Foo"
val myVal : String = "Foo"
```

在 Scala 中声明变量和常量不一定要指明数据类型，在没有指明数据类型的情况下，其数据类型是通过变量或常量的初始值推断出来的。所以，如果在没有指明数据类型的情况下声明变量或常量必须要给出其初始值，否则将会报错。

```
var myVar = 10;
val myVal = "Hello, Scala!";
```

以上实例中，myVar 会被推断为 Int 类型，myVal 会被推断为 String 类型。

Scala 支持多个变量的声明：

```scala
val xmax, ymax = 100  // xmax, ymax都声明为100
```

如果方法返回值是元组，我们可以使用 val 来声明一个元组：

```scala
val pa = (40,"Foo")
pa: (Int, String) = (40,Foo)
```



Scala 访问修饰符基本和Java的一样，分别有：private，protected，public。

如果没有指定访问修饰符，默认情况下，Scala 对象的访问级别都是 public。

Scala 中的 private 限定符，比 Java 更严格，在嵌套类情况下，外层类甚至不能访问被嵌套类的私有成员。

用 private 关键字修饰，带有此标记的成员仅在包含了成员定义的类或对象内部可见，同样的规则还适用内部类。

```scala
class Outer{
    class Inner{
        private def f(){
            println("f")
        }
        class InnerMost{
            f() // 正确
        }
    }
    (new Inner).f() //错误
}
```

在 scala 中，对保护（Protected）成员的访问比 java 更严格一些。因为它只允许保护成员在定义了该成员的的类的子类中被访问。而在java中，用protected关键字修饰的成员，除了定义了该成员的类的子类可以访问，同一个包里的其他类也可以进行访问。

```scala
package p{
class Super{
    protected def f() {println("f")}
    }
        class Sub extends Super{
            f()
        }
        class Other{
                (new Super).f() //错误
        }
}
```

Scala中，访问修饰符可以通过使用限定词强调。格式为:

```scala
private[x] 

protected[x]
```

这里的x指代某个所属的包、类或单例对象。如果写成private[x],读作"这个成员除了对[…]中的类或[…]中的包中的类及它们的伴生对像可见外，对其它所有类都是private。

```scala
package bobsrockets{
    package navigation{
        private[bobsrockets] class Navigator{
         protected[navigation] def useStarChart(){}
         class LegOfJourney{
             private[Navigator] val distance = 100
             }
            private[this] var speed = 200
            }
        }
        package launch{
        import navigation._
        object Vehicle{
        private[launch] val guide = new Navigator
        }
    }
}
```

上述例子中，类 Navigator 被标记为 private[bobsrockets] 就是说这个类对包含在 bobsrockets 包里的所有的类和对象可见。

比如说，从 Vehicle 对象里对 Navigator 的访问是被允许的，因为对象 Vehicle 包含在包 launch 中，而 launch 包在 bobsrockets 中，相反，所有在包 bobsrockets 之外的代码都不能访问类 Navigator。



| 运算符 | 描述     | 实例                      |
| :----- | :------- | :------------------------ |
| ==     | 等于     | (A == B) 运算结果为 false |
| !=     | 不等于   | (A != B) 运算结果为 true  |
| >      | 大于     | (A > B) 运算结果为 false  |
| <      | 小于     | (A < B) 运算结果为 true   |
| >=     | 大于等于 | (A >= B) 运算结果为 false |
| <=     | 小于等于 | (A <= B) 运算结果为 true  |

| 运算符 | 描述   | 实例                       |
| :----- | :----- | :------------------------- |
| &&     | 逻辑与 | (A && B) 运算结果为 false  |
| \|\|   | 逻辑或 | (A \|\| B) 运算结果为 true |
| !      | 逻辑非 | !(A && B) 运算结果为 true  |

```scala
if(布尔表达式 1){
   // 如果布尔表达式 1 为 true 则执行该语句块
}else if(布尔表达式 2){
   // 如果布尔表达式 2 为 true 则执行该语句块
}else if(布尔表达式 3){
   // 如果布尔表达式 3 为 true 则执行该语句块
}else {
   // 如果以上条件都为 false 执行该语句块
}
```

```scala
for( var x <- Range ){
   statement(s);
}
```

**Range** 可以是一个数字区间表示 **i to j** ，或者 **i until j**。左箭头 <- 用于为变量 x 赋值。以下是一个使用了 **i to j** 语法(包含 j)的实例:

```scala
object Test {
   def main(args: Array[String]) {
      var a = 0;
      // for 循环
      for( a <- 1 to 10){
         println( "Value of a: " + a );
      }
   }
}
```

Scala 中使用 **val** 语句可以定义函数，**def** 语句定义方法。

```scala
class Test{
  def m(x: Int) = x + 3
  val f = (x: Int) => x + 3
}
```

Scala 方法声明格式如下：

```scala
def functionName ([参数列表]) : [return type]

def functionName ([参数列表]) : [return type] = {
   function body
   return [expr]
}
object add{
   def addInt( a:Int, b:Int ) : Int = {
      var sum:Int = 0
      sum = a + b

      return sum
   }
}
```

如果方法没有返回值，可以返回为 **Unit**，这个类似于 Java 的 **void**

Scala 提供了多种不同的方法调用方式：以下是调用方法的标准格式：

```scala
functionName( 参数列表 )
//如果方法使用了实例的对象来调用，我们可以使用类似java的格式 (使用 . 号)：
[instance.]functionName( 参数列表 )
```

Scala的解释器在解析函数参数(function arguments)时有两种方式：

- 传值调用（call-by-value）：先计算参数表达式的值，再应用到函数内部；
- 传名调用（call-by-name）：将未计算的参数表达式直接应用到函数内部

在进入函数内部前，传值调用方式就已经将参数表达式的值计算完毕，而传名调用是在函数内部进行参数表达式的值计算的。



一般情况下函数调用参数，就按照函数定义时的参数顺序一个个传递。但是我们也可以通过指定函数参数名，并且不需要按照顺序向函数传递参数

```scala
object Test {
   def main(args: Array[String]) {
        printInt(b=5, a=7);
   }
   def printInt( a:Int, b:Int ) = {
      println("Value of a : " + a );
      println("Value of b : " + b );
   }
}
```

Scala 允许你指明函数的最后一个参数可以是重复的，即我们不需要指定函数参数的个数，可以向函数传入可变长度参数列表。

Scala 通过在参数的类型之后放一个星号来设置可变参数(可重复的参数)。

```scala
object Test {
   def main(args: Array[String]) {
        printStrings("Runoob", "Scala", "Python");
   }
   def printStrings( args:String* ) = {
      var i : Int = 0;
      for( arg <- args ){
         println("Arg value[" + i + "] = " + arg );
         i = i + 1;
      }
   }
}
```

递归函数在函数式编程的语言中起着重要的作用。

Scala 同样支持递归函数。

递归函数意味着函数可以调用它本身。

```scala
object Test {
   def main(args: Array[String]) {
      for (i <- 1 to 10)
         println(i + " 的阶乘为: = " + factorial(i) )
   }
   
   def factorial(n: BigInt): BigInt = {  
      if (n <= 1)
         1  
      else    
      n * factorial(n - 1)
   }
}
```

Scala 可以为函数参数指定默认参数值，使用了默认参数，你在调用函数的过程中可以不需要传递参数，这时函数就会调用它的默认参数值，如果传递了参数，则传递值会取代默认值。

```scala
object Test {
   def main(args: Array[String]) {
        println( "返回值 : " + addInt() );
   }
   def addInt( a:Int=5, b:Int=7 ) : Int = {
      var sum:Int = 0
      sum = a + b

      return sum
   }
}
```

高阶函数（Higher-Order Function）就是操作其他函数的函数。

Scala 中允许使用高阶函数, 高阶函数可以使用其他函数作为参数，或者使用函数作为输出结果。

```scala
object Test {
   def main(args: Array[String]) {

      println( apply( layout, 10) )

   }
   // 函数 f 和 值 v 作为参数，而函数 f 又调用了参数 v
   def apply(f: Int => String, v: Int) = f(v)

   def layout[A](x: A) = "[" + x.toString() + "]"
   
}
```

我们可以在 Scala 函数内定义函数，定义在函数内的函数称之为局部函数。

```scala
object Test {
   def main(args: Array[String]) {
      println( factorial(0) )
      println( factorial(1) )
      println( factorial(2) )
      println( factorial(3) )
   }

   def factorial(i: Int): Int = {
      def fact(i: Int, accumulator: Int): Int = {
         if (i <= 1)
            accumulator
         else
            fact(i - 1, i * accumulator)
      }
      fact(i, 1)
   }
}
```

Scala 中定义匿名函数的语法很简单，箭头左边是参数列表，右边是函数体。

```scala
var inc = (x:Int) => x+1
var mul = (x: Int, y: Int) => x*y
```

```scala
object Test {  
   def main(args: Array[String]) {  
      println( "muliplier(1) value = " +  multiplier(1) )  
      println( "muliplier(2) value = " +  multiplier(2) )  
   }  
   var factor = 3  
   val multiplier = (i:Int) => i * factor  
}
```

###### 数组

```scala
var z:Array[String] = new Array[String](3)

var z = new Array[String](3)
```

z 声明一个字符串类型的数组，数组长度为 3 ，可存储 3 个元素。我们可以为每个元素设置值，并通过索引来访问每个元素

```scala
var z = Array("Runoob", "Baidu", "Google")
```

```scala
val myMatrix = Array.ofDim[Int](3, 3)
```



