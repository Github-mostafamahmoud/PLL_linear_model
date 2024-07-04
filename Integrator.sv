`timescale 1ps/1ps
parameter int n=23;
module Integ_delay(input signed [n:0] I, input CLK , output reg signed [n:0] Out);

reg signed [n:0] Prev;

always @(posedge CLK) begin
#3
Prev = Prev+I;
Out =Prev;

end
initial begin
Prev<=24'd00;
Out<=24'd0000000;
end
endmodule


module Integ(input signed [n:0] I, input CLK , output reg signed [n:0] Out);

reg signed [n:0] Prev;

always @(posedge CLK) begin
#1
Prev = Prev+I;
Out =Prev;

end
initial begin
Prev=24'd0000000;
Out=24'd0000000;
end
endmodule
