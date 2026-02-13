# Hard Cases

This artifact summarizes the lowest-recall classes and representative model failures.

## fully_evasive (selected recall=0.516, f1=0.369)
- true=`fully_evasive` predicted=`intermediate` confidence=`0.852`
  - question: Okay. And as we start -- as we model out our second quarter, you mentioned that B2B is experiencing lower revenue trends in the Q2 '20 and retail is experiencing reduced traffic and lower revenue e...
  - answer: Obviously, we're like -- we can't give specifics on that, but we clearly talked about the fact that there was challenges in the last half of March and early in April, and I think it all depends Liz on what happens from a reopening perspective. And why it's ...
- true=`fully_evasive` predicted=`direct` confidence=`0.782`
  - question: Okay, got it. And then just on that. The wildfire plan you're referring to was that $200 million, I think $70 million of 10-year proposal filed with the commission several months ago, that is still...
  - answer: To be clear there, Brian, we filed with Idaho for Idaho's portion previously, and now we're including the Washington portion here.

## intermediate (selected recall=0.629, f1=0.618)
- true=`intermediate` predicted=`fully_evasive` confidence=`0.923`
  - question: Okay. So you can't comment if it's getting easier or harder to take market share?
  - answer: I mean, right now, I don't think it's currently making too much of a difference. I'm sure that it's not wearing well on our competitors, but, yes, I don't have too much else to say about it.
- true=`intermediate` predicted=`fully_evasive` confidence=`0.905`
  - question: Good morning. This is Lez on for Joon. Thank you for taking my question. Joe, I guess we'll start on the pricing. Thank you for that color that you provided. Can you provide a little bit more detai...
  - answer: Well, look, we've listed WAC as part of our NTAP application. Certainly, we are not bound to list at that WAC. As we said in the prepared comments, we have to set one WAC even though we do expect to have different net pricing across different settings of ca...

## direct (selected recall=0.641, f1=0.669)
- true=`direct` predicted=`fully_evasive` confidence=`0.984`
  - question: OK. Thanks, Rachel. And then maybe just lastly on rivipansel, following the top-line release by Pfizer, do you plan on hosting your own conference call?
  - answer: So again, I don't think the top line release is going to be at a level of detail that's -- we're not going to be -- we're not going to go beyond what's in the press release until there's some opportunity to share a broader data set, as I said, in the contex...
- true=`direct` predicted=`fully_evasive` confidence=`0.948`
  - question: Thank you. Our next question comes from Daina Graybosch from SVB Leerink. Your line is open.
  - answer: Thank you. We do not have a question from Daina Graybosch at this time. We'll move on to the next question.
