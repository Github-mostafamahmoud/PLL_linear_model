`timescale 1ns/1fs
module jiiter_attenuator_tb();
reg input_clk,output_clk;


int N;
jiiter_attenuator attenuator(input_clk,N,output_clk);
always 
#4.5375 input_clk=~input_clk;
initial begin
 input_clk=0;
 
 
N=60;


end
endmodule
