#!/bin/bash
echo "Start Test"
wsk -i action invoke ac --result
wsk -i action invoke sa --result
wsk -i action invoke dt --result



sleep 180
wsk -i action invoke ac --result
wsk -i action invoke sa --result
wsk -i action invoke dt --result


sleep 120
wsk -i action invoke ac --result
wsk -i action invoke sa --result
wsk -i action invoke dt --result

sleep 180
wsk -i action invoke ac --result
wsk -i action invoke sa --result
wsk -i action invoke dt --result

sleep 601
wsk -i action invoke ac --result
wsk -i action invoke sa --result:q
wsk -i action invoke dt --result

sleep 420
wsk -i action invoke ac --result
wsk -i action invoke sa --result
wsk -i action invoke dt --result
echo "Start End"