`timescale 1ns/1fs
module sysDev_tb();
reg input_clk,output_clk;
reg signed[9:0] alpha;


int N;
sysDiv #(.Fref(150e6)) system(input_clk,N,output_clk,alpha);
always 
#3.33333333333333333333333333333 input_clk=~input_clk;
initial begin
 alpha=10'd8;
 input_clk=0;
 
 
N=64;

end
endmodule
//,.Tref(6.666666667e-9)
//3.33333333333333333333333333333