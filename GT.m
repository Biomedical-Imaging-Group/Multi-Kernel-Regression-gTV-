function [f] = GT(t)
%%%GT truth function for our example
% Picewise-linear with three knots at {(0.45,0),(7/15,-2),(8/15,2)}
f=-120*(t-0.45).*(t>0.45)+180*(t-7/15).*(t>7/15)-60*(t-8/15).*(t>8/15);
end

