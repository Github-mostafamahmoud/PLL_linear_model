module DLF #(parameter WIDTH_IN , parameter WIDTH_OUT,parameter real alpha,parameter real beta)
//WIDTH_IN depends on TDC and WIDTH_OUT depends on DAC
// Signals Initiation
(
  input logic clk,
  input logic rst,
  input logic signed [(WIDTH_IN-1):0] IN,
  output logic signed [(WIDTH_OUT-1):0] OUT
); 
  // Parameters Initiations
   //real alpha ;//= 20.51;
   //real beta ;//= 0.48;//0.48;
  reg signed[(WIDTH_IN-1):0] OLD_IN ;
  reg signed[(WIDTH_OUT-1):0] OLD_OUT ;
  logic signed [(WIDTH_OUT-1):0] Max_OUT  ;
  logic signed [(WIDTH_OUT-1):0] Min_OUT  ;
//Maximum Value of y assuming maximum value of beta = 1 , to overcome overflow
  real y ;
 
  initial begin
  OUT=0 ;
  OLD_IN = 0 ;
  OLD_OUT = 0 ;
  y=0;
  Max_OUT = (2**(WIDTH_OUT-1))-1 ;
  Min_OUT = -(2**(WIDTH_OUT-1)) ;
  
  end

//sampling at negative edge to cover all cases of TDC including the case that if Ref is lagging the Feedback. 
  always @(negedge clk or posedge rst) begin


    if (rst==1) begin
        OUT = 0;
    end 

    else begin
        y = OLD_OUT - alpha*OLD_IN + (alpha+beta)*IN ;
        OLD_IN = IN ;
        OLD_OUT = y ;

        // Saturation condition
        if (y >= Max_OUT ) begin
        OUT = Max_OUT ;
        OLD_OUT = Max_OUT ; // to overcome the overflow of y
        end 

        else if (y <= Min_OUT) begin
        OUT = Min_OUT ;
        OLD_OUT = Min_OUT ;
        end

        else begin
        OUT = y ;
        end
     end
  end
endmodule

























